"""Unit tests for the retrieval layer."""

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# BM25
# ──────────────────────────────────────────────────────────────────────────────

def test_bm25_tokenize():
    from app.retrieval.bm25_retrieval import _tokenize

    tokens = _tokenize("Hello World! How are you?")
    assert "hello" in tokens
    assert "world!" in tokens


def test_bm25_returns_empty_on_no_index(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    # Force empty corpus
    monkeypatch.setattr(bm25_mod, "_bm25", None)
    monkeypatch.setattr(bm25_mod, "_corpus", [])

    results = bm25_mod.bm25_search("anything", top_k=5)
    assert results == []


def test_bm25_search_ranks_relevant_higher(monkeypatch):
    from rank_bm25 import BM25Okapi
    import app.retrieval.bm25_retrieval as bm25_mod

    corpus = [
        {"id": 1, "text": "transformer attention mechanism in NLP", "source": "a.pdf"},
        {"id": 2, "text": "random unrelated content about cooking", "source": "b.pdf"},
        {"id": 3, "text": "self-attention and transformer architecture", "source": "c.pdf"},
    ]
    tokenized = [bm25_mod._tokenize(d["text"]) for d in corpus]
    monkeypatch.setattr(bm25_mod, "_corpus", corpus)
    monkeypatch.setattr(bm25_mod, "_bm25", BM25Okapi(tokenized))

    results = bm25_mod.bm25_search("transformer architecture", top_k=3)
    assert len(results) >= 1
    # Cooking doc should score lower than transformer docs
    sources = [r["source"] for r in results]
    if len(results) > 1:
        assert "b.pdf" not in sources[:1]


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid / RRF
# ──────────────────────────────────────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    from app.retrieval.hybrid import _rrf_score

    assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)


def test_hybrid_deduplicates(monkeypatch):
    from app.retrieval import hybrid

    doc = {"id": 42, "text": "shared doc", "source": "x.pdf", "score": 0.9}

    monkeypatch.setattr(hybrid, "dense_search", lambda q, top_k: [doc])
    monkeypatch.setattr(hybrid, "bm25_search", lambda q, top_k: [doc])

    results = hybrid.hybrid_search("test", top_k=5)
    # Same doc should appear only once
    ids = [r["id"] for r in results]
    assert ids.count(42) == 1


def test_hybrid_fuses_scores(monkeypatch):
    from app.retrieval import hybrid

    dense_docs = [
        {"id": 1, "text": "doc one", "source": "a", "score": 0.9},
        {"id": 2, "text": "doc two", "source": "b", "score": 0.8},
    ]
    bm25_docs = [
        {"id": 2, "text": "doc two", "source": "b", "score": 5.0},
        {"id": 3, "text": "doc three", "source": "c", "score": 4.0},
    ]
    monkeypatch.setattr(hybrid, "dense_search", lambda q, top_k: dense_docs)
    monkeypatch.setattr(hybrid, "bm25_search", lambda q, top_k: bm25_docs)

    results = hybrid.hybrid_search("test", top_k=5)
    # doc 2 appears in both sources so should rank highly
    ids = [r["id"] for r in results]
    assert 2 in ids
    assert ids.index(2) == 0  # highest fused score


# ──────────────────────────────────────────────────────────────────────────────
# Dense text_to_id
# ──────────────────────────────────────────────────────────────────────────────

def test_text_to_id_deterministic():
    from app.retrieval.dense import text_to_id

    id1 = text_to_id("hello world", "doc.pdf")
    id2 = text_to_id("hello world", "doc.pdf")
    assert id1 == id2


def test_text_to_id_different_inputs_differ():
    from app.retrieval.dense import text_to_id

    assert text_to_id("hello", "a.pdf") != text_to_id("world", "a.pdf")
