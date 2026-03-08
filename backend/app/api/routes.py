import json
import re

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_initial_state, run_agent, stream_agent_updates
from app.agent.planner import classify_rule_based
from app.database import get_db
from app.ingestion.pipeline import ingest_document, ingest_text_chunks
from app.memory.strategy_memory import load_bandit, log_retrieval_diagnostics, log_run, save_bandit
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
from app.optimization.bandit import CONFIG_NAMES, compute_reward, compute_utility
from app.retrieval.dense import delete_by_document_id, delete_by_source, wipe_all_embeddings

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
        await db.execute(text("SELECT 1"))
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

    query_context = await _prepare_query_context(query, db)

    # Run the agent with learned bandit priors
    final_state = await run_agent(
        query,
        query_context["learned_alpha"],
        query_context["learned_beta"],
        document_ids=query_context["active_doc_ids"],
    )

    return await _finalize_query_response(
        db=db,
        query=query,
        final_state=final_state,
        bandits=query_context["bandits"],
        predicted_type=query_context["predicted_type"],
    )


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_context = await _prepare_query_context(query, db)

    async def event_generator():
        current_state = build_initial_state(
            query,
            query_context["learned_alpha"],
            query_context["learned_beta"],
            document_ids=query_context["active_doc_ids"],
        )

        try:
            async for update in stream_agent_updates(
                query,
                query_context["learned_alpha"],
                query_context["learned_beta"],
                document_ids=query_context["active_doc_ids"],
            ):
                for node_name, payload in update.items():
                    previous_state = dict(current_state)
                    current_state.update(payload)

                    if node_name == "retry":
                        yield _sse_event(
                            {
                                "type": "retrying",
                                "details": {
                                    "retry_count": current_state.get("retry_count", 0),
                                    "previous_config": previous_state.get("current_config", ""),
                                    "selected_config": current_state.get("current_config", ""),
                                    "reason": "faithfulness below threshold",
                                    "failed_faithfulness": previous_state.get("faithfulness", 0.0),
                                },
                            }
                        )
                        continue

                    event = _build_stream_event(node_name, current_state)
                    if event is not None:
                        yield _sse_event(event)

            response = await _finalize_query_response(
                db=db,
                query=query,
                final_state=current_state,
                bandits=query_context["bandits"],
                predicted_type=query_context["predicted_type"],
            )
            yield _sse_event({"type": "query_complete", "response": response.model_dump()})
        except Exception as exc:
            yield _sse_event({"type": "query_error", "error": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _prepare_query_context(query: str, db: AsyncSession) -> dict:
    # Pre-classify query type cheaply (rule-based, no LLM call) so we can
    # load the *correct* bandit and inject learned priors into the agent.
    predicted_type = classify_rule_based(query)
    bandit = await load_bandit(db, predicted_type)

    # Also pre-load the other bandits so we can switch after the agent
    # confirms the real query_type (via its LLM planner).
    bandits = {predicted_type: bandit}
    for qt in ("factual", "comparative", "multi_hop"):
        if qt != predicted_type:
            bandits[qt] = await load_bandit(db, qt)

    doc_result = await db.execute(select(Document.id))
    active_doc_ids = [row[0] for row in doc_result.all()]

    return {
        "predicted_type": predicted_type,
        "bandits": bandits,
        "learned_alpha": dict(bandit.alpha),
        "learned_beta": dict(bandit.beta),
        "active_doc_ids": active_doc_ids,
    }


async def _finalize_query_response(
    db: AsyncSession,
    query: str,
    final_state: dict,
    bandits: dict,
    predicted_type: str,
) -> QueryResponse:
    query_type = final_state["query_type"]
    bandit = bandits.get(query_type, bandits[predicted_type])

    first_config = final_state["first_config"]
    final_config = final_state["current_config"]

    faithfulness = final_state["faithfulness"]
    citation_precision = final_state.get("citation_precision", 0.0)
    unsupported_claim_rate = final_state.get("unsupported_claim_rate", 1.0)
    answer_completeness = final_state.get("answer_completeness", 0.0)
    cost = final_state["cost"]
    latency = final_state["latency"]
    utility = compute_reward(
        faithfulness=faithfulness,
        citation_precision=citation_precision,
        answer_completeness=answer_completeness,
        latency=latency,
    )
    retry_count = final_state["retry_count"]
    retrieval_diagnostics = final_state.get("retrieval_diagnostics", {})

    if retry_count > 0:
        fail_utility = compute_utility(faithfulness * 0.3, cost * 0.6, latency * 0.6)
        bandit.update(first_config, fail_utility)
        await log_run(
            db, query, query_type, first_config,
            faithfulness * 0.3, cost * 0.6, latency * 0.6, fail_utility,
            is_retry=False,
        )

    bandit.update(final_config, utility)
    await log_run(db, query, query_type, final_config, faithfulness, cost, latency, utility, is_retry=retry_count > 0)
    await log_retrieval_diagnostics(db, query, query_type, final_config, retrieval_diagnostics)
    await save_bandit(db, query_type, bandit)

    docs = final_state["all_docs"]
    citations = _extract_citations(final_state["answer"], docs)

    return QueryResponse(
        answer=final_state["answer"],
        citations=citations,
        metrics=RunMetrics(
            faithfulness=round(faithfulness, 4),
            citation_precision=round(citation_precision, 4),
            unsupported_claim_rate=round(unsupported_claim_rate, 4),
            answer_completeness=round(answer_completeness, 4),
            cost=round(cost, 6),
            latency=round(latency, 2),
            utility=round(utility, 4),
            config=final_config,
            query_type=query_type,
            hops=final_state["hop"],
            retrieval_diagnostics=retrieval_diagnostics,
        ),
        abstained=final_state.get("abstained", False),
    )


def _build_stream_event(node_name: str, state: dict) -> dict | None:
    if node_name == "plan":
        return {
            "type": "step_completed",
            "step": "planner",
            "details": {
                "query_type": state.get("query_type"),
                "selected_config": state.get("current_config"),
            },
        }

    if node_name == "query_rewrite":
        return {
            "type": "step_completed",
            "step": "query_rewriting",
            "details": {
                "variants_generated": len(state.get("query_variants", [])),
                "primary_query": (state.get("query_variants") or [state.get("current_query", "")])[0],
            },
        }

    if node_name == "retrieve":
        diagnostics = state.get("retrieval_diagnostics", {})
        return {
            "type": "step_completed",
            "step": "retrieval",
            "details": {
                "documents_retrieved": len(state.get("all_docs", [])),
                "query_coverage": diagnostics.get("query_coverage", 0.0),
                "document_diversity": diagnostics.get("document_diversity", 0.0),
            },
        }

    if node_name == "read":
        return {
            "type": "step_completed",
            "step": "reader",
            "details": {
                "evidence_spans": len(state.get("evidence", [])),
                "followup_query": state.get("followup_query") or "none",
            },
        }

    if node_name == "controller":
        diagnostics = state.get("retrieval_diagnostics", {})
        return {
            "type": "step_completed",
            "step": "research_controller",
            "details": {
                "evidence_coverage": state.get("evidence_coverage", 0.0),
                "recall_proxy": diagnostics.get("estimated_recall_proxy", 0.0),
                "decision": state.get("controller_action", "stop"),
            },
        }

    if node_name == "write":
        return {
            "type": "step_completed",
            "step": "writer",
            "details": {
                "answer_length": len(state.get("answer", "")),
                "abstained": state.get("abstained", False),
            },
        }

    if node_name == "claim_extract":
        return {
            "type": "step_completed",
            "step": "claim_extraction",
            "details": {
                "claims_extracted": len(state.get("claims", [])),
            },
        }

    if node_name == "citation_verify":
        return {
            "type": "step_completed",
            "step": "citation_verification",
            "details": {
                "citation_precision": state.get("citation_precision", 0.0),
                "unsupported_claim_rate": state.get("unsupported_claim_rate", 0.0),
            },
        }

    if node_name == "evaluate":
        return {
            "type": "step_completed",
            "step": "evaluation",
            "details": {
                "faithfulness": state.get("faithfulness", 0.0),
                "answer_completeness": state.get("answer_completeness", 0.0),
                "confidence": state.get("evaluator_confidence", 0.0),
            },
        }

    return None


def _extract_citations(answer: str, docs: list[dict]) -> list[CitedChunk]:
    """Extract [N] citation references from the answer and map to source docs."""
    cited_indices = set()
    # Find all brackets containing numbers, potentially separated by commas/spaces
    # e.g., [1], [1, 2], [1, 2, 3]
    for m in re.finditer(r"\[([\d\s,]+)\]", answer):
        content = m.group(1)
        # Split by comma and clean up
        parts = content.split(",")
        for p in parts:
            try:
                idx = int(p.strip())
                if 1 <= idx <= len(docs):
                    cited_indices.add(idx)
            except ValueError:
                continue

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
    """Ingest each user-selected paper as a separate document."""
    if not body.papers:
        raise HTTPException(status_code=400, detail="No papers selected.")

    # Create separate records for each paper
    doc_tasks = []
    for p in body.papers:
        doc = Document(
            filename=p.title,
            source=p.url or p.title,
            status="processing"
        )
        db.add(doc)
        doc_tasks.append((doc, p.dict()))
    
    await db.commit()
    
    # After commit, docs have IDs
    tasks_info = [(d.id, p_dict) for d, p_dict in doc_tasks]

    async def run_individual_ingest(task_list: list[tuple[int, dict]]):
        import httpx
        from app.database import async_session_factory
        from app.ingestion.pipeline import ingest_document

        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            for doc_id, p in task_list:
                async with async_session_factory() as session:
                    try:
                        pdf_content = None
                        filename = f"{p['title']}.pdf"
                        
                        # Try to get full PDF if arXiv URL is available
                        arxiv_url = p.get("arxiv_url")
                        if arxiv_url and "arxiv.org/abs/" in arxiv_url:
                            pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"
                            try:
                                resp = await client.get(pdf_url)
                                if resp.status_code == 200:
                                    pdf_content = resp.content
                            except Exception as download_err:
                                pass

                        if pdf_content:
                            # Ingest as full document (PDF)
                            count = ingest_document(filename, pdf_content, document_id=doc_id)
                        else:
                            # Fallback: Ingest abstract only as text
                            text = f"Title: {p['title']}\n\nAbstract: {p['abstract']}"
                            if p.get("author"):
                                text += f"\n\nAuthor: {p['author']}"
                            if p.get("published"):
                                text += f"\nPublished: {p['published']}"
                            
                            source = p.get("url") or p["title"]
                            count = ingest_text_chunks(source, [text], document_id=doc_id)
                        
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

    background_tasks.add_task(run_individual_ingest, tasks_info)

    return DocumentUploadResponse(
        message=f"Ingestion started for {len(body.papers)} selected papers. Full PDFs will be downloaded where possible.",
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
    # Also delete by source as a backup (important for legacy data)
    delete_by_source(doc.source)
    
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted successfully"}


@router.post("/documents/wipe")
async def wipe_all_documents(db: AsyncSession = Depends(get_db)):
    """Wipe all documents from the database and all embeddings from Qdrant."""
    from sqlalchemy import delete
    
    # 1. Clear database
    await db.execute(delete(Document))
    await db.commit()
    
    # 2. Clear Qdrant collection
    wipe_all_embeddings()
    
    return {"message": "All database records and vector embeddings have been wiped."}


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
