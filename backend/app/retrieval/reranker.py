import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


class Reranker:
    async def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        if not docs:
            return docs
        loop = asyncio.get_event_loop()
        pairs = [(query, d["text"]) for d in docs]
        reranker = get_reranker()
        scores = await loop.run_in_executor(None, reranker.predict, pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        reranked = sorted(docs, key=lambda d: d.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]
