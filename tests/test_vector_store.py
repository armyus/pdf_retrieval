"""
Tests for InMemoryVectorStore (V2 — Vector Store).
"""

import numpy as np
import pytest

from src.chunker import TextChunk
from src.vector_store import InMemoryVectorStore, ScoredChunk


class TestInMemoryVectorStore:

    def test_empty_store(self):
        store = InMemoryVectorStore()
        assert store.count == 0
        results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
        assert results == []

    def test_add_and_search_top_k(self):
        store = InMemoryVectorStore()

        chunk1 = TextChunk("c1", "f1.pdf", "/f1.pdf", 1, "chunk 1")
        chunk2 = TextChunk("c2", "f2.pdf", "/f2.pdf", 2, "chunk 2")
        chunk3 = TextChunk("c3", "f3.pdf", "/f3.pdf", 3, "chunk 3")

        # Normalized vectors:
        # v1 aligned with [1, 0]
        # v2 orthogonal [0, 1]
        # v3 semi-aligned [0.7071, 0.7071]
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ], dtype=np.float32)

        store.add([chunk1, chunk2, chunk3], embeddings)
        assert store.count == 3

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "c1"
        assert pytest.approx(results[0].score, rel=1e-3) == 1.0
        assert results[1].chunk.chunk_id == "c3"
        assert pytest.approx(results[1].score, rel=1e-3) == 0.7071

    def test_score_threshold_filter(self):
        store = InMemoryVectorStore()
        chunk1 = TextChunk("c1", "f1.pdf", "/f1.pdf", 1, "chunk 1")
        chunk2 = TextChunk("c2", "f2.pdf", "/f2.pdf", 2, "chunk 2")

        embeddings = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float32)

        store.add([chunk1, chunk2], embeddings)

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=5, score_threshold=0.5)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_mismatch_exception(self):
        store = InMemoryVectorStore()
        chunks = [TextChunk("c1", "f1.pdf", "/f1.pdf", 1, "chunk 1")]
        embeddings = np.zeros((2, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="Mismatch between chunks count"):
            store.add(chunks, embeddings)

    def test_clear(self):
        store = InMemoryVectorStore()
        chunks = [TextChunk("c1", "f1.pdf", "/f1.pdf", 1, "chunk 1")]
        embeddings = np.ones((1, 2), dtype=np.float32)

        store.add(chunks, embeddings)
        assert store.count == 1

        store.clear()
        assert store.count == 0
        assert store.embeddings is None
