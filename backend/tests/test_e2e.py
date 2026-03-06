"""Integration tests for the end-to-end query pipeline."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# Skip integration tests if no API key is set
SKIP_INTEGRATION = not os.getenv("GEMINI_API_KEY")


@pytest.fixture
async def test_app():
    """Create a test FastAPI app with mocked dependencies."""
    from app.main import app
    from app.api import routes

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        return_value={
            "query": "Why did Transformers replace RNNs?",
            "answer": "Transformers replaced RNNs due to parallelization and attention [1].",
            "citations": [{"doc_id": "d1", "title": "Attention is All You Need", "span": "attention mechanisms", "score": 0.9}],
            "query_type": "factual",
            "config_id": "A",
            "faithfulness": 0.85,
            "citation_grounding": 0.80,
            "utility": 0.72,
            "cost": 0.00012,
            "latency": 4.2,
            "retry_count": 0,
        }
    )
    mock_agent.bandit = MagicMock()
    mock_agent.bandit.get_arm_stats = MagicMock(
        return_value={
            "A": {"count": 1, "mean_reward": 0.72, "ucb1_score": 1.5, "top_k": 5, "chunk_size": 200, "rerank": False},
            "B": {"count": 0, "mean_reward": 0.0, "ucb1_score": float("inf"), "top_k": 8, "chunk_size": 300, "rerank": False},
            "C": {"count": 0, "mean_reward": 0.0, "ucb1_score": float("inf"), "top_k": 8, "chunk_size": 300, "rerank": True},
        }
    )

    await routes.set_agent(mock_agent)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=1)
    mock_session.execute = AsyncMock(return_value=AsyncMock(scalar_one_or_none=lambda: None))

    # Mock async_session_factory used by the background _log() task
    mock_sf = MagicMock()
    mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.api.routes.get_session", return_value=mock_session), \
         patch("app.api.routes.async_session_factory", mock_sf):
        yield app


@pytest.mark.asyncio
async def test_health_endpoint(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_endpoint_returns_structured_response(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/query",
            json={"query": "Why did Transformers replace RNNs?"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "faithfulness" in data
    assert "utility" in data
    assert "latency" in data
    assert "cost" in data
    assert data["config_id"] in ("A", "B", "C")


@pytest.mark.asyncio
async def test_query_rejects_empty_string(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingestion_pipeline_unit():
    """Unit test ingestion without Qdrant/BM25 side effects."""
    from app.ingestion.pipeline import _chunk_text, _hierarchical_chunks

    # chunk_text basic
    text = "word " * 100
    chunks = _chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 60  # chunk_size + small buffer

    # hierarchical_chunks
    md = "# Section 1\n\nFirst paragraph text here.\n\n## Section 2\n\nSecond paragraph text."
    chunks = _hierarchical_chunks(md, chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert "text" in c
        assert "section" in c
