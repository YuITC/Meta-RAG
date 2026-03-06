"""Unit tests for the retrieval pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval.bm25_retrieval import BM25Retriever
from app.retrieval.hybrid import HybridRetriever


class TestBM25Retriever:
    def test_search_returns_empty_when_no_corpus(self, tmp_path):
        retriever = BM25Retriever(data_dir=str(tmp_path))
        results = retriever.search("transformer attention", top_k=3)
        assert results == []

    def test_add_and_search(self, tmp_path):
        retriever = BM25Retriever(data_dir=str(tmp_path))
        docs = [
            "Transformers use self-attention mechanisms",
            "RNNs process sequences step by step",
            "Attention allows parallel computation",
        ]
        ids = ["doc1", "doc2", "doc3"]
        retriever.add_documents(ids, docs)

        results = retriever.search("attention transformer", top_k=2)
        assert len(results) <= 2
        assert all("id" in r and "text" in r and "score" in r for r in results)
        # Top result should be about attention/transformers
        assert results[0]["id"] in ["doc1", "doc3"]

    def test_corpus_persistence(self, tmp_path):
        r1 = BM25Retriever(data_dir=str(tmp_path))
        r1.add_documents(["a"], ["hello world"])

        r2 = BM25Retriever(data_dir=str(tmp_path))
        results = r2.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_no_duplicate_ids(self, tmp_path):
        retriever = BM25Retriever(data_dir=str(tmp_path))
        retriever.add_documents(["dup"], ["first text"])
        retriever.add_documents(["dup"], ["second text - should not be added"])
        retriever._load_corpus()
        assert retriever._doc_ids.count("dup") == 1


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_merges_and_deduplicates(self):
        retriever = HybridRetriever()

        doc_a = {"id": "doc1", "score": 0.9, "text": "A" * 400, "title": "T", "source": "s"}
        doc_b = {"id": "doc2", "score": 0.7, "text": "B" * 100, "title": "T", "source": "s"}
        doc_dup = {"id": "doc1", "score": 0.8, "text": "A" * 400, "title": "T", "source": "s"}

        retriever.dense.search = AsyncMock(return_value=[doc_a, doc_b])
        retriever.bm25.search = MagicMock(return_value=[doc_dup])

        results = await retriever.retrieve("test query", config_id="A")

        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"
        assert len(results) <= 5  # top_k for config A

    @pytest.mark.asyncio
    async def test_truncation_applied(self):
        retriever = HybridRetriever()
        long_text = "word " * 500  # much longer than chunk_size=200

        retriever.dense.search = AsyncMock(
            return_value=[{"id": "x", "score": 0.9, "text": long_text, "title": "", "source": ""}]
        )
        retriever.bm25.search = MagicMock(return_value=[])

        results = await retriever.retrieve("query", config_id="A")  # chunk_size=200
        assert len(results[0]["text"]) <= 200 * 4 + 10  # character-level truncation
