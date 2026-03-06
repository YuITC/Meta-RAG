import json
import os
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Retriever:
    """BM25 retrieval over a corpus stored as a JSON file."""

    def __init__(self, data_dir: str | None = None):
        _dir = data_dir if data_dir is not None else settings.data_dir
        self.corpus_path = Path(_dir) / "bm25_corpus.json"
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _load_corpus(self) -> None:
        if self.corpus_path.exists():
            with open(self.corpus_path) as f:
                data = json.load(f)
            self._doc_ids = data.get("doc_ids", [])
            self._texts = data.get("texts", [])
        else:
            self._doc_ids = []
            self._texts = []

    def _save_corpus(self) -> None:
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "w") as f:
            json.dump({"doc_ids": self._doc_ids, "texts": self._texts}, f)

    def _build_index(self) -> None:
        if not self._texts:
            self._bm25 = None
            return
        tokenized = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def add_documents(self, doc_ids: list[str], texts: list[str]) -> None:
        self._load_corpus()
        existing_ids = set(self._doc_ids)
        for doc_id, text in zip(doc_ids, texts):
            if doc_id not in existing_ids:
                self._doc_ids.append(doc_id)
                self._texts.append(text)
        self._save_corpus()
        self._build_index()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._bm25 is None:
            self._load_corpus()
            self._build_index()
        if self._bm25 is None or not self._texts:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # rank_bm25 can produce negative IDF scores for small corpora (all docs contain
        # the term), so we don't filter by absolute score — we just return the top-k ranked docs.
        return [
            {
                "id": self._doc_ids[idx],
                "score": float(scores[idx]),
                "text": self._texts[idx],
                "title": "",
                "source": "bm25",
                "chunk_index": idx,
            }
            for idx in top_indices
        ]

    def rebuild(self, doc_ids: list[str], texts: list[str]) -> None:
        """Rebuild corpus from scratch (e.g., after loading from Qdrant)."""
        self._doc_ids = doc_ids
        self._texts = texts
        self._save_corpus()
        self._build_index()
