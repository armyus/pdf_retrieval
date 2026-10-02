"""
Tests for Modular Retrievers (V5 — Swappable Retrievers & Hybrid Fusion).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pdf_parser import PageContent, ParsedPDF
from src.chunker import TextChunk
from src.vector_store import ScoredChunk
from src.config import RetrievalConfig
from src.retrievers import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    KeywordRetriever,
    get_retriever,
)


@pytest.fixture
def mock_documents():
    doc1 = ParsedPDF(
        filename="cryptography.pdf",
        full_path="/docs/cryptography.pdf",
        pages=[
            PageContent(page_number=1, text="RSA encryption uses prime numbers for confidentiality and public key distribution."),
        ],
        total_pages=1,
    )
    doc2 = ParsedPDF(
        filename="biology.pdf",
        full_path="/docs/biology.pdf",
        pages=[
            PageContent(page_number=1, text="Photosynthesis converts solar light into biochemical plant energy."),
        ],
        total_pages=1,
    )
    return [doc1, doc2]


class TestModularRetrievers:

    def test_factory_get_retriever(self):
        assert isinstance(get_retriever("hybrid"), HybridRetriever)
        assert isinstance(get_retriever("dense"), DenseRetriever)
        assert isinstance(get_retriever("bm25"), BM25Retriever)
        assert isinstance(get_retriever("keyword"), KeywordRetriever)

        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            get_retriever("invalid_strategy")

    def test_bm25_retriever(self, mock_documents):
        retriever = BM25Retriever()
        count = retriever.index_documents(mock_documents)
        assert count > 0

        results = retriever.search("prime numbers RSA", top_k=2)
        assert len(results) > 0
        assert results[0].chunk.filename == "cryptography.pdf"

    def test_keyword_retriever(self, mock_documents):
        retriever = KeywordRetriever()
        retriever.index_documents(mock_documents)

        results = retriever.search("Photosynthesis", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.filename == "biology.pdf"

    def test_hybrid_retriever_reciprocal_rank_fusion(self):
        config = RetrievalConfig(fusion_method="rrf", rrf_k=60, dense_weight=0.6, sparse_weight=0.4)
        hybrid = HybridRetriever(config=config)

        chunk1 = TextChunk("c1", "a.pdf", "/a.pdf", 1, "text 1")
        chunk2 = TextChunk("c2", "b.pdf", "/b.pdf", 1, "text 2")

        # Dense: c1 rank 1, c2 rank 2
        dense_res = [ScoredChunk(chunk1, 0.9), ScoredChunk(chunk2, 0.5)]
        # BM25: c2 rank 1, c1 rank 2
        bm25_res = [ScoredChunk(chunk2, 5.0), ScoredChunk(chunk1, 2.0)]

        fused = hybrid._reciprocal_rank_fusion(dense_res, bm25_res, top_k=2)

        assert len(fused) == 2
        # c1 rrf: 0.6/(60+1) + 0.4/(60+2) = 0.6/61 + 0.4/62 = 0.009836 + 0.006451 = 0.016287
        # c2 rrf: 0.6/(60+2) + 0.4/(60+1) = 0.6/62 + 0.4/61 = 0.009677 + 0.006557 = 0.016234
        # c1 should win slightly due to higher dense_weight (0.6 vs 0.4)
        assert fused[0].chunk.chunk_id == "c1"

    def test_hybrid_retriever_weighted_score_fusion(self):
        config = RetrievalConfig(fusion_method="weighted", dense_weight=0.5, sparse_weight=0.5)
        hybrid = HybridRetriever(config=config)

        chunk1 = TextChunk("c1", "a.pdf", "/a.pdf", 1, "text 1")
        chunk2 = TextChunk("c2", "b.pdf", "/b.pdf", 1, "text 2")

        dense_res = [ScoredChunk(chunk1, 1.0), ScoredChunk(chunk2, 0.0)]
        bm25_res = [ScoredChunk(chunk1, 10.0), ScoredChunk(chunk2, 0.0)]

        fused = hybrid._weighted_score_fusion(dense_res, bm25_res, top_k=2)
        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "c1"
        assert fused[0].score > fused[1].score
