"""
Keyword Search — V1: Simple keyword-based search across parsed PDFs.

Searches for a query string across all pages of all parsed documents
and returns matching passages with context.
"""

import re
from dataclasses import dataclass
from typing import List

from src.pdf_parser import ParsedPDF


@dataclass
class SearchResult:
    filename: str
    full_path: str
    page_number: int
    snippet: str
    match_count: int        # number of times query appears on this page


class KeywordSearcher:
    """Case-insensitive keyword search across parsed PDFs."""

    def __init__(self, context_chars: int = 120):
        """
        Args:
            context_chars: Number of characters of context to show
                           around each match in the snippet.
        """
        self.context_chars = context_chars

    def search(
        self,
        documents: List[ParsedPDF],
        query: str,
    ) -> List[SearchResult]:
        """
        Search all documents for the query string.

        Performs case-insensitive substring matching.
        Returns results sorted by match count (highest first).

        Args:
            documents: List of parsed PDF documents.
            query: The keyword or phrase to search for.

        Returns:
            List of SearchResult objects, one per page that contains a match.
        """
        if not query or not query.strip():
            return []

        query_clean = query.strip()
        pattern = re.compile(re.escape(query_clean), re.IGNORECASE)

        results: List[SearchResult] = []

        for doc in documents:
            if doc.error:
                continue

            for page in doc.pages_with_text:
                matches = list(pattern.finditer(page.text))
                if not matches:
                    continue

                # Build snippet from the first match with surrounding context
                snippet = self._extract_snippet(page.text, matches[0], self.context_chars)

                results.append(SearchResult(
                    filename=doc.filename,
                    full_path=doc.full_path,
                    page_number=page.page_number,
                    snippet=snippet,
                    match_count=len(matches),
                ))

        # Sort: most matches first, then by filename, then page number
        results.sort(key=lambda r: (-r.match_count, r.filename.lower(), r.page_number))
        return results

    @staticmethod
    def _extract_snippet(text: str, match: re.Match, context_chars: int) -> str:
        """Extract a snippet of text around a regex match."""
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)

        snippet = text[start:end].strip()
        # Clean up excessive whitespace
        snippet = re.sub(r"\s+", " ", snippet)

        # Add ellipsis markers
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        # Sanitize for console output (replace unencodable chars)
        snippet = snippet.encode("ascii", errors="replace").decode("ascii")

        return snippet


def print_search_results(results: List[SearchResult], query: str) -> None:
    """Pretty-print keyword search results."""
    if not results:
        print(f'\nNo results found for: "{query}"')
        return

    count = len(results)
    label = "result" if count == 1 else "results"
    print(f'\nFound {count} {label} for: "{query}"\n')
    print("-" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result.filename}")
        print(f"   Page: {result.page_number}")
        print(f"   Matches: {result.match_count}")
        print(f'   "{result.snippet}"')
        print()
