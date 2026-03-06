import asyncio

from app.config import RETRIEVAL_CONFIGS
from app.retrieval.dense import DenseRetriever
from app.retrieval.bm25_retrieval import BM25Retriever
from app.retrieval.reranker import Reranker


def _truncate_text(text: str, chunk_size: int) -> str:
    """Approximate character-level truncation to respect chunk_size."""
    return text[:chunk_size * 4] if len(text) > chunk_size * 4 else text


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()

    async def retrieve(self, query: str, config_id: str) -> list[dict]:
        cfg = RETRIEVAL_CONFIGS.get(config_id, RETRIEVAL_CONFIGS["A"])
        top_k: int = cfg["top_k"]
        chunk_size: int = cfg["chunk_size"]
        do_rerank: bool = cfg["rerank"]

        # Run dense and BM25 in parallel
        dense_results, bm25_results = await asyncio.gather(
            self.dense.search(query, top_k=top_k),
            asyncio.get_event_loop().run_in_executor(
                None, self.bm25.search, query, top_k
            ),
        )

        # Merge and deduplicate by doc id
        seen: set[str] = set()
        merged: list[dict] = []
        for doc in dense_results + bm25_results:
            if doc["id"] not in seen:
                seen.add(doc["id"])
                doc["text"] = _truncate_text(doc["text"], chunk_size)
                merged.append(doc)

        if do_rerank and merged:
            merged = await self.reranker.rerank(query, merged, top_k=top_k)
        else:
            # Sort by score descending, take top_k
            merged = sorted(merged, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

        return merged
