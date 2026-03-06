from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.agent.graph import ResearchAgent
from app.ingestion.pipeline import IngestionPipeline
from app.api import routes as api_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    await init_db()

    agent = ResearchAgent()
    await agent.initialize()
    pipeline = IngestionPipeline()

    await api_routes.set_agent(agent)
    await api_routes.set_ingestion(pipeline)

    yield

    # Shutdown (nothing to clean up currently)


app = FastAPI(
    title="Autonomous Research Agent",
    description="RAG system with adaptive retrieval optimization via multi-armed bandit",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_routes.router, prefix="/api/v1")
