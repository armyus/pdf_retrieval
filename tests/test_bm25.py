"""
Tests for BM25 Engine (V5 — Sparse BM25Okapi Lexical Search).
"""

from src.chunker import TextChunk
from src.bm25 import BM25Index, tokenize


class TestBM25Index:

    def test_tokenize(self):
        text = "Hello, World! Transformers & Neural-Networks 123."
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "transformers", "neural", "networks", "123"]

    def test_empty_index_and_query(self):
        index = BM25Index()
        assert index.total_docs == 0
        assert index.search("query") == []

        chunk = TextChunk("c1", "a.pdf", "/a.pdf", 1, "Some text")
        index.index([chunk])
        assert index.search("") == []
        assert index.search("   ") == []

    def test_bm25_ranking(self):
        index = BM25Index()

        chunks = [
            TextChunk("c1", "ml.pdf", "/ml.pdf", 1, "Machine learning algorithms build models from sample training data."),
            TextChunk("c2", "crypto.pdf", "/crypto.pdf", 1, "RSA encryption uses prime numbers for cryptographic security."),
            TextChunk("c3", "mixed.pdf", "/mixed.pdf", 1, "Algorithms in cryptography and machine learning."),
        ]

        index.index(chunks)
        assert index.total_docs == 3

        # Query "cryptographic security"
        results = index.search("cryptographic security", top_k=2)

        assert len(results) > 0
        # c2 should rank first because it contains both cryptographic and security
        assert results[0].chunk.chunk_id == "c2"
        assert results[0].score > 0.0

    def test_term_frequency_saturation(self):
        index = BM25Index(k1=1.5, b=0.75)

        # Doc 1 has 1 occurrence, Doc 2 has 5 occurrences
        chunk1 = TextChunk("c1", "a.pdf", "/a.pdf", 1, "quantum quantum quantum quantum quantum physics")
        chunk2 = TextChunk("c2", "b.pdf", "/b.pdf", 1, "quantum physics and chemistry")

        index.index([chunk1, chunk2])
        results = index.search("quantum")

        assert len(results) == 2
        # Doc 1 with higher TF should score higher
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].score > results[1].score
