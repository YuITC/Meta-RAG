import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session, async_session_factory
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    Citation,
    IngestResponse,
    StatsResponse,
)
from app.models.db_models import QueryLog, BanditArm
from app.agent.graph import ResearchAgent
from app.ingestion.pipeline import IngestionPipeline
from app.config import settings

router = APIRouter()

# Singletons (initialized on startup via lifespan)
_agent: ResearchAgent | None = None
_ingestion: IngestionPipeline | None = None


def get_agent() -> ResearchAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


def get_ingestion() -> IngestionPipeline:
    if _ingestion is None:
        raise HTTPException(status_code=503, detail="Ingestion not initialized")
    return _ingestion


async def set_agent(agent: ResearchAgent) -> None:
    global _agent
    _agent = agent


async def set_ingestion(pipeline: IngestionPipeline) -> None:
    global _ingestion
    _ingestion = pipeline


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    agent: ResearchAgent = Depends(get_agent),
) -> QueryResponse:
    """Run a query through the research agent."""
    result = await agent.run(request.query)

    # Persist query log in background using its own session (request session is closed by then)
    async def _log():
        async with async_session_factory() as session:
            session.add(
                QueryLog(
                    query=request.query,
                    query_type=result.get("query_type", ""),
                    config_id=result.get("config_id", ""),
                    faithfulness=result.get("faithfulness", 0.0),
                    citation_grounding=result.get("citation_grounding", 0.0),
                    utility=result.get("utility", 0.0),
                    cost=result.get("cost", 0.0),
                    latency=result.get("latency", 0.0),
                    answer=result.get("answer", ""),
                    retry_count=result.get("retry_count", 0),
                )
            )
            await session.commit()

    background_tasks.add_task(_log)

    citations = [
        Citation(
            doc_id=c.get("doc_id", ""),
            title=c.get("title", ""),
            text=c.get("span", ""),
            score=c.get("rerank_score", c.get("score", 0.0)),
        )
        for c in (result.get("citations") or [])
    ]

    return QueryResponse(
        query=request.query,
        answer=result.get("answer", ""),
        citations=citations,
        query_type=result.get("query_type", ""),
        config_id=result.get("config_id", ""),
        faithfulness=result.get("faithfulness", 0.0),
        citation_grounding=result.get("citation_grounding", 0.0),
        utility=result.get("utility", 0.0),
        cost=result.get("cost", 0.0),
        latency=result.get("latency", 0.0),
        retry_count=result.get("retry_count", 0),
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    pipeline: IngestionPipeline = Depends(get_ingestion),
) -> IngestResponse:
    """Upload and ingest a document into the vector store."""
    suffix = Path(file.filename).suffix
    tmp_path = Path(settings.data_dir) / f"upload_{int(time.time())}{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        n_chunks = await pipeline.ingest(
            str(tmp_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        file_path=file.filename, chunks_indexed=n_chunks, status="success"
    )


@router.post("/ingest/path", response_model=IngestResponse)
async def ingest_path(
    file_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    pipeline: IngestionPipeline = Depends(get_ingestion),
) -> IngestResponse:
    """Ingest a document by file system path (server-side)."""
    n_chunks = await pipeline.ingest(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return IngestResponse(file_path=file_path, chunks_indexed=n_chunks, status="success")


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    agent: ResearchAgent = Depends(get_agent),
    session: AsyncSession = Depends(get_session),
) -> StatsResponse:
    """Return bandit arm statistics and query metrics."""
    total = await session.scalar(select(func.count(QueryLog.id)))
    avg_utility = await session.scalar(select(func.avg(QueryLog.utility)))
    avg_latency = await session.scalar(select(func.avg(QueryLog.latency)))
    avg_cost = await session.scalar(select(func.avg(QueryLog.cost)))

    return StatsResponse(
        bandit_arms=agent.bandit.get_arm_stats(),
        total_queries=total or 0,
        avg_utility=round(avg_utility, 4) if avg_utility else None,
        avg_latency=round(avg_latency, 2) if avg_latency else None,
        avg_cost=round(avg_cost, 6) if avg_cost else None,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
