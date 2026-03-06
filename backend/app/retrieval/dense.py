import asyncio
from functools import lru_cache

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> list[float]:
    embedder = get_embedder()
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()


class DenseRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection = settings.qdrant_collection
        self._initialized = False

    async def ensure_collection(self) -> None:
        if self._initialized:
            return
        embedder = get_embedder()
        dim = embedder.get_sentence_embedding_dimension()
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection not in names:
            await self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._initialized = True

    async def upsert(self, points: list[dict]) -> None:
        """points: list of {id, vector, payload}"""
        await self.ensure_collection()
        structs = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        await self.client.upsert(self.collection, points=structs)

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        await self.ensure_collection()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, embed_text, query)
        results = await self.client.search(
            self.collection,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "title": r.payload.get("title", ""),
                "source": r.payload.get("source", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
            }
            for r in results
        ]

    async def get_all_texts(self) -> tuple[list[str], list[str]]:
        """Return (ids, texts) for BM25 index building."""
        await self.ensure_collection()
        ids, texts = [], []
        offset = None
        while True:
            records, next_offset = await self.client.scroll(
                self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                ids.append(str(r.id))
                texts.append(r.payload.get("text", ""))
            if next_offset is None:
                break
            offset = next_offset
        return ids, texts
