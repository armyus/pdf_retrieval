"""
Tests for the PDF Parser module (V1 — Text Extraction).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pdf_parser import PageContent, ParsedPDF, PDFParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return PDFParser()


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal valid PDF for testing.

    Uses PyMuPDF itself to create a real PDF with known text content
    so we can verify extraction.
    """
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, this is a test PDF document.")
    page.insert_text((72, 100), "It contains important information about testing.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two has different content about algorithms.")

    pdf_path = tmp_path / "test_sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def empty_pdf(tmp_path):
    """Create a PDF with no text content."""
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()  # blank page
    pdf_path = tmp_path / "empty.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


# ---------------------------------------------------------------------------
# Tests: PageContent
# ---------------------------------------------------------------------------

class TestPageContent:

    def test_char_count_computed(self):
        page = PageContent(page_number=1, text="Hello world")
        assert page.char_count == 11

    def test_has_text_true(self):
        page = PageContent(page_number=1, text="Some text")
        assert page.has_text is True

    def test_has_text_false_empty(self):
        page = PageContent(page_number=1, text="")
        assert page.has_text is False

    def test_has_text_false_whitespace(self):
        page = PageContent(page_number=1, text="   \n\t  ")
        assert page.has_text is False


# ---------------------------------------------------------------------------
# Tests: ParsedPDF
# ---------------------------------------------------------------------------

class TestParsedPDF:

    def test_has_text_true(self):
        doc = ParsedPDF(
            filename="test.pdf",
            full_path="/a/test.pdf",
            pages=[PageContent(1, "hello"), PageContent(2, "")],
            total_pages=2,
        )
        assert doc.has_text is True

    def test_has_text_false(self):
        doc = ParsedPDF(
            filename="test.pdf",
            full_path="/a/test.pdf",
            pages=[PageContent(1, ""), PageContent(2, "  ")],
            total_pages=2,
        )
        assert doc.has_text is False

    def test_full_text(self):
        doc = ParsedPDF(
            filename="test.pdf",
            full_path="/a/test.pdf",
            pages=[PageContent(1, "hello"), PageContent(2, "world")],
            total_pages=2,
        )
        assert "hello" in doc.full_text
        assert "world" in doc.full_text

    def test_pages_with_text_filters(self):
        doc = ParsedPDF(
            filename="test.pdf",
            full_path="/a/test.pdf",
            pages=[PageContent(1, "hello"), PageContent(2, ""), PageContent(3, "world")],
            total_pages=3,
        )
        assert len(doc.pages_with_text) == 2

    def test_error_doc(self):
        doc = ParsedPDF(
            filename="bad.pdf",
            full_path="/a/bad.pdf",
            error="File not found",
        )
        assert doc.error is not None
        assert doc.has_text is False


# ---------------------------------------------------------------------------
# Tests: PDFParser.parse
# ---------------------------------------------------------------------------

class TestPDFParserParse:

    def test_parse_valid_pdf(self, parser, sample_pdf):
        result = parser.parse(sample_pdf)
        assert result.error is None
        assert result.total_pages == 2
        assert result.has_text is True
        assert "test PDF document" in result.full_text

    def test_parse_second_page(self, parser, sample_pdf):
        result = parser.parse(sample_pdf)
        page2 = result.pages[1]
        assert page2.page_number == 2
        assert "algorithms" in page2.text

    def test_parse_empty_pdf(self, parser, empty_pdf):
        result = parser.parse(empty_pdf)
        assert result.error is None
        assert result.total_pages == 1
        assert result.has_text is False

    def test_parse_nonexistent_file(self, parser):
        result = parser.parse("/nonexistent/fake.pdf")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_parse_non_pdf_file(self, parser, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("just a text file")
        result = parser.parse(str(txt_file))
        assert result.error is not None
        assert "not a pdf" in result.error.lower()

    def test_parse_corrupted_pdf(self, parser, tmp_path):
        bad_pdf = tmp_path / "corrupted.pdf"
        bad_pdf.write_bytes(b"this is not a real pdf file at all")
        result = parser.parse(str(bad_pdf))
        # Should either error gracefully or have no text
        # PyMuPDF may or may not throw — either way should not crash
        assert result is not None

    def test_full_path_is_absolute(self, parser, sample_pdf):
        result = parser.parse(sample_pdf)
        assert os.path.isabs(result.full_path)

    def test_filename_extracted(self, parser, sample_pdf):
        result = parser.parse(sample_pdf)
        assert result.filename == "test_sample.pdf"


# ---------------------------------------------------------------------------
# Tests: PDFParser.parse_many
# ---------------------------------------------------------------------------

class TestPDFParserParseMany:

    def test_parse_many(self, parser, sample_pdf, empty_pdf):
        results = parser.parse_many([sample_pdf, empty_pdf])
        assert len(results) == 2
        assert results[0].has_text is True
        assert results[1].has_text is False

    def test_parse_many_empty_list(self, parser):
        results = parser.parse_many([])
        assert results == []

    def test_parse_many_with_bad_path(self, parser, sample_pdf):
        results = parser.parse_many([sample_pdf, "/fake/missing.pdf"])
        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error is not None


# ---------------------------------------------------------------------------
# Tests: Real data files (integration)
# ---------------------------------------------------------------------------

class TestRealDataIntegration:
    """Test against the actual PDFs in the data/ folder."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    @pytest.fixture
    def real_pdfs(self):
        """Get paths to real PDFs if they exist."""
        data = Path(self.DATA_DIR).resolve()
        if not data.exists():
            pytest.skip("data/ folder not found")
        pdfs = list(data.glob("*.pdf")) + list(data.glob("*.PDF"))
        if not pdfs:
            pytest.skip("No PDFs in data/ folder")
        return [str(p) for p in pdfs]

    def test_real_pdfs_parse_without_crash(self, real_pdfs):
        parser = PDFParser()
        for path in real_pdfs:
            result = parser.parse(path)
            assert result is not None
            assert result.error is None, f"Failed to parse {path}: {result.error}"

    def test_real_pdfs_have_text(self, real_pdfs):
        parser = PDFParser()
        texts = []
        for path in real_pdfs:
            result = parser.parse(path)
            if result.has_text:
                texts.append(result.filename)
        assert len(texts) > 0, "Expected at least one PDF with extractable text"
