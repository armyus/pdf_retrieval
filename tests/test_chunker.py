"""
Tests for TextChunker (V2 — Text Chunking).
"""

import pytest
from src.chunker import TextChunk, TextChunker
from src.pdf_parser import PageContent, ParsedPDF


class TestTextChunker:

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            TextChunker(chunk_size=100, chunk_overlap=-1)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_short_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_empty_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n\t  ") == []

    def test_chunk_overlapping_text(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        # text length 35 chars
        text = "12345678901234567890123456789012345"
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2
        # Verify first chunk length <= 20
        assert len(chunks[0]) <= 20

    def test_chunk_page(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        page = PageContent(page_number=3, text="This is a test paragraph designed to test chunking functionality across pages.")
        chunks = chunker.chunk_page(page, filename="test.pdf", full_path="/data/test.pdf")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.filename == "test.pdf"
            assert chunk.full_path == "/data/test.pdf"
            assert chunk.page_number == 3
            assert chunk.chunk_id.startswith("test.pdf_p3_c")
            assert len(chunk.text) > 0

    def test_chunk_page_empty(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        page = PageContent(page_number=1, text="   ")
        chunks = chunker.chunk_page(page, filename="test.pdf", full_path="/data/test.pdf")
        assert chunks == []

    def test_chunk_document(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        doc = ParsedPDF(
            filename="paper.pdf",
            full_path="/path/paper.pdf",
            pages=[
                PageContent(page_number=1, text="First page content."),
                PageContent(page_number=2, text="Second page content with more words for chunking."),
            ],
            total_pages=2,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2
        assert any(c.page_number == 1 for c in chunks)
        assert any(c.page_number == 2 for c in chunks)

    def test_chunk_document_with_error(self):
        chunker = TextChunker()
        doc = ParsedPDF(
            filename="error.pdf",
            full_path="/path/error.pdf",
            error="Corrupted file",
        )
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_documents_multiple(self):
        chunker = TextChunker()
        doc1 = ParsedPDF(filename="doc1.pdf", full_path="/doc1.pdf", pages=[PageContent(1, "Text one.")], total_pages=1)
        doc2 = ParsedPDF(filename="doc2.pdf", full_path="/doc2.pdf", pages=[PageContent(1, "Text two.")], total_pages=1)
        chunks = chunker.chunk_documents([doc1, doc2])
        assert len(chunks) == 2
        assert chunks[0].filename == "doc1.pdf"
        assert chunks[1].filename == "doc2.pdf"
