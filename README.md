# Autonomous Research Agent

A self-optimizing RAG (Retrieval-Augmented Generation) pipeline that classifies queries, runs hybrid retrieval (dense + BM25), evaluates its own answers, and uses Thompson Sampling to adapt retrieval configurations over time.

## Features

- **Adaptive Retrieval** — Thompson Sampling bandit selects the best retrieval configuration per query type, learning from each run
- **Multi-Hop Reasoning** — automatically performs follow-up retrieval when evidence is insufficient
- **Hybrid Search** — fuses dense (BGE embeddings) and sparse (BM25) retrieval with Reciprocal Rank Fusion
- **Self-Evaluation** — single-call LLM evaluator scores faithfulness, completeness, and confidence after every answer
- **Automatic Retry** — re-runs the pipeline with a different configuration if faithfulness falls below threshold
- **Citation Verification** — extracts claims and verifies each is supported by retrieved evidence
- **Evidence Provenance Graph** — tracks which claims are supported by which documents
- **Research Controller** — decides whether to stop, do another hop, reformulate, or abstain based on coverage signals
- **Retrieval Guardrails** — filters prompt injection and low-quality chunks before they reach the LLM
- **Streaming Pipeline** — SSE-based real-time updates showing each pipeline stage as it completes

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Pipeline                              │
│                                                                        │
│  plan → query_rewrite → retrieve → read → controller ─┐               │
│                              ▲                         │               │
│                              └── (extra_hop/reform.) ──┘               │
│                                                        │               │
│                                                        ▼               │
│  write → claim_extract → citation_verify → evidence_graph → evaluate   │
│                                                              │         │
│                                              (retry if low) ─┘         │
└────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Nodes

| Node | Description |
|------|-------------|
| **Planner** | Classifies query as factual / comparative / multi_hop; selects retrieval config via Thompson Sampling |
| **Query Rewriter** | Generates search variants (template-based or LLM-based) for better recall |
| **Retriever** | Hybrid dense + BM25 search with optional cross-encoder reranking |
| **Reader** | Extracts evidence spans and identifies missing information |
| **Controller** | Decides stop / extra_hop / reformulate / abstain based on coverage signals |
| **Writer** | Generates grounded Markdown answer with inline citations |
| **Claim Extractor** | Splits answer into atomic claims |
| **Citation Verifier** | Checks each claim against retrieved documents |
| **Evidence Graph** | Builds provenance links from claims to source documents |
| **Evaluator** | Scores faithfulness, completeness, and confidence in a single LLM call |

### Retrieval Configurations

The bandit selects from four configurations, each with different cost/quality tradeoffs:

| Config | top_k | chunk_size | rewrite | rerank | hop_limit |
|--------|-------|------------|---------|--------|-----------|
| A | 5 | 256 | None | No | 1 |
| B | 10 | 512 | Template (3) | No | 1 |
| C | 10 | 512 | Template (4) | Yes | 2 |
| D | 8 | 512 | LLM (5) | No | 2 |

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemini 2.5 Flash (langchain-google-genai) |
| Agent | LangGraph |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL 16 |
| Embeddings | BGE-small-en-v1.5 (sentence-transformers) |
| Reranker | BGE-reranker-base (cross-encoder) |
| Backend | FastAPI + uvicorn |
| Frontend | Next.js 15 + Tailwind CSS |
| State | Zustand |
| Tests | pytest + pytest-asyncio |

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 20+
- Docker & Docker Compose
- A [Gemini API key](https://ai.google.dev/)

### 1. Clone and configure

```bash
git clone https://github.com/<your-username>/autonomous-research-agent.git
cd autonomous-research-agent
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up -d qdrant postgres
```

### 3. Run the backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Docker (full stack)

```bash
docker compose up -d
```

This starts Qdrant, PostgreSQL, the API server, and the frontend.

## Running Tests

```bash
cd backend
uv run pytest
```

## Data Sources

- **File upload** — PDF, HTML, Markdown, DOCX, TXT via the API or UI
- **HuggingFace Trending** — scrapes the top 50 trending papers and optionally downloads full arXiv PDFs

## Project Structure

```
backend/
  app/
    agent/       # LangGraph pipeline (planner, reader, writer, graph)
    api/         # FastAPI routes
    ingestion/   # Document parsing and chunking
    memory/      # Bandit state persistence
    models/      # Pydantic schemas + SQLAlchemy models
    optimization/# Thompson Sampling bandit + evaluator
    research/    # Controller, coverage, evidence graph
    retrieval/   # Dense, BM25, hybrid, reranker, guardrails
    verification/# Claim extraction + citation verification
  scraper/       # HuggingFace trending paper scraper
  tests/         # Unit and integration tests
frontend/
  src/
    app/         # Next.js pages
    components/  # React components
    lib/         # API client, types, state store
```

## License

MIT
