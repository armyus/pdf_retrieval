"""
Tests for the Keyword Search module (V1 — Keyword Search).
"""

import pytest

from src.pdf_parser import PageContent, ParsedPDF
from src.keyword_search import KeywordSearcher, SearchResult, print_search_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def searcher():
    return KeywordSearcher(context_chars=50)


@pytest.fixture
def sample_documents():
    """Create mock ParsedPDF documents for searching."""
    doc1 = ParsedPDF(
        filename="transformers.pdf",
        full_path="/papers/transformers.pdf",
        pages=[
            PageContent(
                page_number=1,
                text="The transformer architecture uses self-attention mechanisms.",
            ),
            PageContent(
                page_number=2,
                text="Convolutional networks are used for image recognition.",
            ),
            PageContent(
                page_number=3,
                text="The attention mechanism allows the model to focus on relevant parts.",
            ),
        ],
        total_pages=3,
    )

    doc2 = ParsedPDF(
        filename="biology.pdf",
        full_path="/papers/biology.pdf",
        pages=[
            PageContent(
                page_number=1,
                text="Photosynthesis converts sunlight into chemical energy.",
            ),
            PageContent(
                page_number=2,
                text="The cell membrane controls what enters and exits the cell.",
            ),
        ],
        total_pages=2,
    )

    doc3 = ParsedPDF(
        filename="failed.pdf",
        full_path="/papers/failed.pdf",
        error="Could not open file",
    )

    return [doc1, doc2, doc3]


# ---------------------------------------------------------------------------
# Tests: KeywordSearcher.search
# ---------------------------------------------------------------------------

class TestKeywordSearch:

    def test_basic_search(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "attention")
        filenames = [r.filename for r in results]
        assert "transformers.pdf" in filenames
        assert "biology.pdf" not in filenames

    def test_finds_multiple_pages(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "attention")
        # Should match page 1 ("self-attention") and page 3 ("attention mechanism")
        assert len(results) == 2
        pages = sorted([r.page_number for r in results])
        assert pages == [1, 3]

    def test_case_insensitive(self, searcher, sample_documents):
        results_lower = searcher.search(sample_documents, "the")
        results_upper = searcher.search(sample_documents, "THE")
        assert len(results_lower) == len(results_upper)

    def test_no_results(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "quantum computing")
        assert results == []

    def test_empty_query(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "")
        assert results == []

    def test_whitespace_query(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "   ")
        assert results == []

    def test_skips_error_documents(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "could not")
        # "failed.pdf" has an error, so it should be skipped entirely
        error_results = [r for r in results if r.filename == "failed.pdf"]
        assert len(error_results) == 0

    def test_match_count(self, searcher):
        doc = ParsedPDF(
            filename="repeats.pdf",
            full_path="/papers/repeats.pdf",
            pages=[
                PageContent(1, "test test test and more test here"),
            ],
            total_pages=1,
        )
        results = searcher.search([doc], "test")
        assert len(results) == 1
        assert results[0].match_count == 4

    def test_sorting_by_match_count(self, searcher):
        doc = ParsedPDF(
            filename="mixed.pdf",
            full_path="/papers/mixed.pdf",
            pages=[
                PageContent(1, "word word word"),       # 3 matches
                PageContent(2, "word"),                 # 1 match
                PageContent(3, "word word word word"),  # 4 matches
            ],
            total_pages=3,
        )
        results = searcher.search([doc], "word")
        assert results[0].match_count == 4  # highest first
        assert results[-1].match_count == 1  # lowest last

    def test_snippet_contains_query(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "photosynthesis")
        assert len(results) == 1
        assert "photosynthesis" in results[0].snippet.lower()

    def test_cross_document_search(self, searcher, sample_documents):
        results = searcher.search(sample_documents, "the")
        filenames = set(r.filename for r in results)
        # Should find "the" in both transformers.pdf and biology.pdf
        assert len(filenames) >= 2

    def test_empty_documents_list(self, searcher):
        results = searcher.search([], "query")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:

    def test_attributes(self):
        result = SearchResult(
            filename="test.pdf",
            full_path="/a/test.pdf",
            page_number=5,
            snippet="...some text...",
            match_count=3,
        )
        assert result.filename == "test.pdf"
        assert result.page_number == 5
        assert result.match_count == 3


# ---------------------------------------------------------------------------
# Tests: print_search_results
# ---------------------------------------------------------------------------

class TestPrintSearchResults:

    def test_no_results_message(self, capsys):
        print_search_results([], "missing term")
        captured = capsys.readouterr()
        assert "No results found" in captured.out
        assert "missing term" in captured.out

    def test_results_printed(self, capsys):
        results = [
            SearchResult("a.pdf", "/a.pdf", 1, "...match here...", 2),
            SearchResult("b.pdf", "/b.pdf", 3, "...another match...", 1),
        ]
        print_search_results(results, "test")
        captured = capsys.readouterr()
        assert "Found 2 results" in captured.out
        assert "a.pdf" in captured.out
        assert "b.pdf" in captured.out

    def test_singular_result(self, capsys):
        results = [SearchResult("a.pdf", "/a.pdf", 1, "...", 1)]
        print_search_results(results, "test")
        captured = capsys.readouterr()
        assert "Found 1 result for" in captured.out
