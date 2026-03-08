from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str


class CitedChunk(BaseModel):
    index: int
    text: str
    source: str


class RunMetrics(BaseModel):
    faithfulness: float
    citation_precision: float = 0.0
    unsupported_claim_rate: float = 1.0
    answer_completeness: float = 0.0
    cost: float
    latency: float
    utility: float
    config: str
    query_type: str
    hops: int
    retrieval_diagnostics: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitedChunk]
    metrics: RunMetrics
    abstained: bool = False


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_indexed: int


class ConfigStats(BaseModel):
    alpha: float
    beta: float
    win_rate: float
    trials: int


class BanditStats(BaseModel):
    query_type: str
    configs: dict[str, ConfigStats]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    database: bool


class DocumentRead(BaseModel):
    id: int
    filename: str
    source: str
    status: str
    chunks_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class PaperInfo(BaseModel):
    title: str
    url: str
    arxiv_url: str | None = None
    github_url: str | None = None
    github_stars: str | None = None
    author: str | None = None
    published: str | None = None
    abstract: str


class IngestSelectedRequest(BaseModel):
    papers: list[PaperInfo]
