from pydantic_settings import BaseSettings, SettingsConfigDict


RETRIEVAL_CONFIGS = {
    "A": {"top_k": 5, "chunk_size": 200, "rerank": False},
    "B": {"top_k": 8, "chunk_size": 300, "rerank": False},
    "C": {"top_k": 8, "chunk_size": 300, "rerank": True},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Look for .env in the current directory AND one level up (project root),
        # so the server works whether invoked from backend/ or the project root.
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "research_papers"

    # Database
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # Embedding & Reranker
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-small"

    # Self-improvement
    faithfulness_threshold: float = 0.7
    lambda_cost: float = 0.3
    lambda_latency: float = 0.1

    # Constraints
    max_latency_seconds: float = 15.0
    max_hops: int = 2

    # Data directory
    data_dir: str = "./data"


settings = Settings()
