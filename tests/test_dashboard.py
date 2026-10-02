"""
Tests for Web Dashboard & REST API (V6 Dashboard).
"""

from unittest.mock import patch, MagicMock
import pytest

from src.dashboard import create_app, STATE
from src.pdf_parser import PageContent, ParsedPDF
from src.chunker import TextChunk
from src.vector_store import ScoredChunk


@pytest.fixture
def mock_app():
    # Setup mock documents
    doc = ParsedPDF(
        filename="test_crypto.pdf",
        full_path="/docs/test_crypto.pdf",
        pages=[PageContent(page_number=1, text="RSA encryption uses prime numbers.")],
        total_pages=1,
    )

    with patch("src.dashboard.PDFScanner") as mock_scanner_cls, \
         patch("src.dashboard.PDFParser") as mock_parser_cls:

        mock_scanner = MagicMock()
        mock_pdf = MagicMock()
        mock_pdf.full_path = "/docs/test_crypto.pdf"
        mock_pdf.filename = "test_crypto.pdf"
        mock_scanner.scan.return_value = [mock_pdf]
        mock_scanner_cls.return_value = mock_scanner

        mock_parser = MagicMock()
        mock_parser.parse_many.return_value = [doc]
        mock_parser_cls.return_value = mock_parser

        app = create_app("./dummy_folder")
        app.config["TESTING"] = True
        return app


class TestDashboardAPI:

    def test_index_page(self, mock_app):
        client = mock_app.test_client()
        response = client.get("/")
        assert response.status_code == 200
        assert b"Personal PDF Knowledge Retrieval" in response.data

    def test_status_endpoint(self, mock_app):
        client = mock_app.test_client()
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "documents_count" in data
        assert "available_strategies" in data

    def test_search_endpoint(self, mock_app):
        client = mock_app.test_client()
        response = client.post("/api/search", json={
            "query": "RSA encryption",
            "strategy": "bm25",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_rag_endpoint(self, mock_app):
        client = mock_app.test_client()
        response = client.post("/api/rag", json={
            "query": "What does RSA encryption use?",
            "strategy": "bm25",
            "top_k": 2,
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "answer" in data
        assert "sources" in data
