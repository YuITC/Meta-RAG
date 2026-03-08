from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # LLM
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://ara:ara_secret@localhost:5432/ara_db"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "research_papers"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Self-improvement thresholds
    faithfulness_threshold: float = 0.7
    lambda_cost: float = 0.3
    lambda_latency: float = 0.2

    # Constraints
    max_latency_seconds: float = 15.0

    # Retrieval intelligence / control
    default_rewrite_count: int = 4
    evidence_coverage_threshold: float = 0.45
    min_retrieval_diversity: float = 0.25
    evaluator_confidence_threshold: float = 0.4

    # Multi-metric reward weights
    reward_w1_faithfulness: float = 0.45
    reward_w2_citation_precision: float = 0.3
    reward_w3_answer_completeness: float = 0.2
    reward_w4_latency_penalty: float = 0.15


settings = Settings()
