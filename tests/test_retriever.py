"""
Tests for SemanticRetriever (V2 — Semantic Search Pipeline).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pdf_parser import PageContent, ParsedPDF
from src.retriever import SemanticRetriever, print_semantic_results


class TestSemanticRetriever:

    @pytest.fixture
    def mock_documents(self):
        doc1 = ParsedPDF(
            filename="ml_intro.pdf",
            full_path="/docs/ml_intro.pdf",
            pages=[
                PageContent(page_number=1, text="Machine learning algorithms build mathematical models based on sample training data."),
                PageContent(page_number=2, text="Supervised learning involves learning a function that maps inputs to outputs."),
            ],
            total_pages=2,
        )
        doc2 = ParsedPDF(
            filename="gardening.pdf",
            full_path="/docs/gardening.pdf",
            pages=[
                PageContent(page_number=1, text="Tomatoes require full sunlight and well-drained fertile soil to thrive."),
            ],
            total_pages=1,
        )
        return [doc1, doc2]

    def test_init_state(self):
        retriever = SemanticRetriever()
        assert not retriever.is_indexed
        assert retriever.total_chunks == 0
        assert retriever.search("query") == []

    def test_index_and_search_mocked(self, mock_documents):
        retriever = SemanticRetriever()

        # Mock embedder to avoid loading torch models during fast unit tests
        retriever.embedder._load_model = MagicMock()

        def fake_embed_texts(texts, batch_size=32, normalize=True):
            # Create synthetic embeddings
            dim = 4
            arr = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                if "machine learning" in text.lower() or "supervised" in text.lower():
                    arr[i] = [1.0, 0.0, 0.0, 0.0]
                else:
                    arr[i] = [0.0, 1.0, 0.0, 0.0]
            return arr

        def fake_embed_query(query, normalize=True):
            if "machine" in query.lower():
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        retriever.embedder.embed_texts = MagicMock(side_effect=fake_embed_texts)
        retriever.embedder.embed_query = MagicMock(side_effect=fake_embed_query)

        count = retriever.index_documents(mock_documents)
        assert count > 0
        assert retriever.is_indexed
        assert retriever.total_chunks == count

        # Query machine learning
        results = retriever.search("machine learning concepts", top_k=2)
        assert len(results) > 0
        assert results[0].chunk.filename == "ml_intro.pdf"
        assert results[0].score > 0.9

    def test_print_semantic_results(self, capsys):
        from src.chunker import TextChunk
        from src.vector_store import ScoredChunk

        scored = [
            ScoredChunk(
                chunk=TextChunk("c1", "test.pdf", "/test.pdf", 1, "This is snippet text"),
                score=0.8876,
            )
        ]
        print_semantic_results(scored, "test query")
        out = capsys.readouterr().out
        assert "Found 1" in out and "semantic result" in out
        assert "test.pdf" in out
        assert "0.8876" in out

    def test_print_empty_results(self, capsys):
        print_semantic_results([], "empty query")
        out = capsys.readouterr().out
        assert "No semantic results found" in out
