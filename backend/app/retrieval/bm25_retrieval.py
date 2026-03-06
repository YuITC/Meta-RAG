from rank_bm25 import BM25Okapi

from app.config import settings

_bm25: BM25Okapi | None = None
_corpus: list[dict] | None = None  # [{id, text, source}]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index() -> None:
    """Rebuild BM25 index by scrolling all Qdrant points."""
    global _bm25, _corpus
    from app.retrieval.dense import get_qdrant_client

    client = get_qdrant_client()
    docs = []
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        docs.extend(result)
        if offset is None:
            break

    if not docs:
        _bm25 = None
        _corpus = []
        return

    _corpus = [
        {"id": d.id, "text": d.payload["text"], "source": d.payload["source"]}
        for d in docs
    ]
    tokenized = [_tokenize(d["text"]) for d in _corpus]
    _bm25 = BM25Okapi(tokenized)


def invalidate_index() -> None:
    """Call after upsert to force rebuild on next search."""
    global _bm25, _corpus
    _bm25 = None
    _corpus = None


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    if _bm25 is None or _corpus is None:
        build_bm25_index()

    if not _bm25 or not _corpus:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]
    return [
        {
            "id": _corpus[i]["id"],
            "text": _corpus[i]["text"],
            "source": _corpus[i]["source"],
            "score": float(scores[i]),
        }
        for i in top_indices
        if scores[i] > 0
    ]
