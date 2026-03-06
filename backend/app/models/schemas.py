from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1, max_length=2000)


class Citation(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    query_type: str
    config_id: str
    faithfulness: float
    citation_grounding: float
    utility: float
    cost: float
    latency: float
    retry_count: int


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to file to ingest")
    chunk_size: int = Field(default=300, ge=100, le=1000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)


class IngestResponse(BaseModel):
    file_path: str
    chunks_indexed: int
    status: str


class StatsResponse(BaseModel):
    bandit_arms: dict[str, dict]
    total_queries: int
    avg_utility: Optional[float]
    avg_latency: Optional[float]
    avg_cost: Optional[float]
