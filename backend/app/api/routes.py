import re

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import run_agent
from app.database import get_db
from app.ingestion.pipeline import ingest_document, ingest_text_chunks
from app.memory.strategy_memory import load_bandit, log_run, save_bandit
from app.models.db_models import Document
from app.models.schemas import (
    BanditStats,
    CitedChunk,
    ConfigStats,
    DocumentRead,
    DocumentUploadResponse,
    HealthResponse,
    IngestSelectedRequest,
    PaperInfo,
    QueryRequest,
    QueryResponse,
    RunMetrics,
)
from app.optimization.bandit import compute_utility
from app.retrieval.dense import delete_by_document_id

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".docx", ".txt"}


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    qdrant_ok = False
    db_ok = False

    try:
        from app.retrieval.dense import get_qdrant_client
        get_qdrant_client().get_collections()
        qdrant_ok = True
    except Exception:
        pass

    try:
        await db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if qdrant_ok and db_ok else "degraded",
        qdrant=qdrant_ok,
        database=db_ok,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Query
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Load bandit — we don't know query_type yet; use a temp "unknown" type
    # We'll reload after plan_node sets query_type. For now load all three.
    # The bandit is keyed by query_type which is determined inside the agent.
    # Strategy: run agent with default priors, then swap to the correct bandit.
    # Better: pre-load all three bandits and inject their combined alpha/beta.

    # Load per-type bandits into a merged lookup
    bandits = {}
    for qt in ("factual", "comparative", "multi_hop"):
        bandits[qt] = await load_bandit(db, qt)

    # Use a placeholder bandit for the initial config selection (factual is default)
    # The graph's plan_node will set query_type; for the first config selection
    # we pass a neutral Beta(1,1) — the correct bandit is applied after we know the type.
    # This is a two-pass approach: plan first to get query_type, then select config.

    # Pass a unified alpha/beta that the graph will use internally.
    # We'll inject the correct bandit alpha/beta after knowing the query type.
    # For simplicity: run with neutral priors, extract query_type, then re-run
    # with the correct bandit. Instead, integrate bandit loading into the graph
    # by using a post-plan hook.

    # Practical solution: run agent with neutral priors, then load the correct
    # bandit AFTER we know the query_type and use it to update.
    neutral_alpha = {"A": 1.0, "B": 1.0, "C": 1.0}
    neutral_beta = {"A": 1.0, "B": 1.0, "C": 1.0}

    # Run the agent
    final_state = await run_agent(query, neutral_alpha, neutral_beta)

    query_type = final_state["query_type"]
    bandit = bandits.get(query_type, bandits["factual"])

    # Recompute config choice with correct bandit for config selection was already done
    # We simply use the config_name from the agent's run.
    first_config = final_state["first_config"]
    final_config = final_state["current_config"]

    faithfulness = final_state["faithfulness"]
    cost = final_state["cost"]
    latency = final_state["latency"]
    utility = final_state["utility"]
    retry_count = final_state["retry_count"]

    # Update bandit: if there was a retry, log both runs
    if retry_count > 0:
        # First run failed — log with low utility
        fail_utility = compute_utility(faithfulness * 0.3, cost * 0.6, latency * 0.6)
        bandit.update(first_config, fail_utility)
        await log_run(
            db, query, query_type, first_config,
            faithfulness * 0.3, cost * 0.6, latency * 0.6, fail_utility,
            is_retry=False,
        )
    # Log final run
    bandit.update(final_config, utility)
    await log_run(db, query, query_type, final_config, faithfulness, cost, latency, utility, is_retry=retry_count > 0)
    await save_bandit(db, query_type, bandit)

    # Parse citations from answer
    docs = final_state["all_docs"]
    citations = _extract_citations(final_state["answer"], docs)

    return QueryResponse(
        answer=final_state["answer"],
        citations=citations,
        metrics=RunMetrics(
            faithfulness=round(faithfulness, 4),
            cost=round(cost, 6),
            latency=round(latency, 2),
            utility=round(utility, 4),
            config=final_config,
            query_type=query_type,
            hops=final_state["hop"],
        ),
    )


def _extract_citations(answer: str, docs: list[dict]) -> list[CitedChunk]:
    """Extract [N] citation references from the answer and map to source docs."""
    cited_indices = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(docs):
            cited_indices.add(idx)

    citations = []
    for idx in sorted(cited_indices):
        doc = docs[idx - 1]
        citations.append(
            CitedChunk(
                index=idx,
                text=doc["text"][:300],
                source=doc["source"],
            )
        )
    return citations


# ──────────────────────────────────────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentRead])
async def get_documents(db: AsyncSession = Depends(get_db)):
    """List all documents in the system."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Document upload
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=DocumentUploadResponse)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    import os

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    # Create document record
    doc = Document(
        filename=file.filename,
        source=file.filename,
        status="processing"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    async def run_ingestion(doc_id: int, filename: str, file_content: bytes):
        from app.database import async_session_factory
        async with async_session_factory() as session:
            try:
                count = ingest_document(filename, file_content, document_id=doc_id)
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "indexed"
                    db_doc.chunks_count = count
                    await session.commit()
            except Exception as e:
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "failed"
                    db_doc.error_message = str(e)
                    await session.commit()

    background_tasks.add_task(run_ingestion, doc.id, file.filename, content)

    return DocumentUploadResponse(
        message=f"Upload started for '{file.filename}'. It will be processed in the background.",
        chunks_indexed=0,
    )


@router.post("/ingest/scrape", response_model=DocumentUploadResponse)
async def ingest_scrape_endpoint(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Scrape trending HuggingFace papers and ingest abstracts."""
    
    # Create a placeholder record for the scrape job
    doc = Document(
        filename="HuggingFace Trending Scrape",
        source="huggingface",
        status="processing"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    async def run_scrape(doc_id: int):
        from app.database import async_session_factory
        from scraper.scraper import fetchTrendingPapers
        async with async_session_factory() as session:
            try:
                papers = fetchTrendingPapers()
                total = 0
                for paper in papers:
                    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
                    if paper.get("author"):
                        text += f"\n\nAuthor: {paper['author']}"
                    if paper.get("published"):
                        text += f"\nPublished: {paper['published']}"
                    source = paper.get("url") or paper["title"]
                    total += ingest_text_chunks(source, [text], document_id=doc_id)
                
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "indexed"
                    db_doc.chunks_count = total
                    await session.commit()
            except Exception as e:
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "failed"
                    db_doc.error_message = str(e)
                    await session.commit()

    background_tasks.add_task(run_scrape, doc.id)

    return DocumentUploadResponse(
        message="Scrape job started. HuggingFace trending papers are being indexed in the background.",
        chunks_indexed=0,
    )


@router.get("/ingest/trending", response_model=list[PaperInfo])
async def get_trending_papers():
    """Fetch the list of 50 trending papers without ingesting them."""
    from scraper.scraper import fetchTrendingPapers
    try:
        papers = fetchTrendingPapers()
        return [PaperInfo(**p) for p in papers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch papers: {str(e)}")


@router.post("/ingest/selected", response_model=DocumentUploadResponse)
async def ingest_selected_papers(
    body: IngestSelectedRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Ingest a user-selected list of papers."""
    if not body.papers:
        raise HTTPException(status_code=400, detail="No papers selected.")

    # Create a generic record for this batch
    batch_name = f"Selected HF Papers (x{len(body.papers)})"
    doc = Document(
        filename=batch_name,
        source="huggingface-selected",
        status="processing"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    async def run_selective_ingest(doc_id: int, papers_to_ingest: list[dict]):
        from app.database import async_session_factory
        async with async_session_factory() as session:
            try:
                total = 0
                for p in papers_to_ingest:
                    text = f"Title: {p['title']}\n\nAbstract: {p['abstract']}"
                    if p.get("author"):
                        text += f"\n\nAuthor: {p['author']}"
                    if p.get("published"):
                        text += f"\nPublished: {p['published']}"
                    source = p.get("url") or p["title"]
                    total += ingest_text_chunks(source, [text], document_id=doc_id)
                
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "indexed"
                    db_doc.chunks_count = total
                    await session.commit()
            except Exception as e:
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "failed"
                    db_doc.error_message = str(e)
                    await session.commit()

    background_tasks.add_task(run_selective_ingest, doc.id, [p.dict() for p in body.papers])

    return DocumentUploadResponse(
        message=f"Ingestion started for {len(body.papers)} selected papers.",
        chunks_indexed=0,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a document and its associated vector embeddings."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from Qdrant using the unified document_id
    delete_by_document_id(doc.id)
    
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted successfully"}


# ──────────────────────────────────────────────────────────────────────────────
# Bandit stats
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/bandit/{query_type}", response_model=BanditStats)
async def bandit_stats(query_type: str, db: AsyncSession = Depends(get_db)):
    if query_type not in ("factual", "comparative", "multi_hop"):
        raise HTTPException(status_code=400, detail="query_type must be factual, comparative, or multi_hop")
    bandit = await load_bandit(db, query_type)
    raw = bandit.stats()
    return BanditStats(
        query_type=query_type,
        configs={
            name: ConfigStats(**stats) for name, stats in raw.items()
        },
    )


@router.get("/bandit", response_model=list[BanditStats])
async def all_bandit_stats(db: AsyncSession = Depends(get_db)):
    result = []
    for qt in ("factual", "comparative", "multi_hop"):
        bandit = await load_bandit(db, qt)
        raw = bandit.stats()
        result.append(BanditStats(
            query_type=qt,
            configs={name: ConfigStats(**stats) for name, stats in raw.items()},
        ))
    return result
