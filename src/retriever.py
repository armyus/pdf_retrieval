"""
Retriever — Unified Retriever Module

Exposes swappable retrievers (Hybrid, Dense, BM25, Keyword)
and formatting utilities for search results.
"""

from typing import List
import re
from collections import defaultdict

from src.retrievers import (
    BaseRetriever,
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    KeywordRetriever,
    get_retriever,
)
from src.vector_store import ScoredChunk

# Explicit SemanticRetriever alias to DenseRetriever
SemanticRetriever = DenseRetriever


def print_semantic_results(
    results: List[ScoredChunk],
    query: str,
    show_metadata: bool = True,
    group_by_doc: bool = False,
) -> None:
    """Pretty-print retrieval search results with hierarchical metadata."""
    if not results:
        print(f'\nNo semantic results found for: "{query}"')
        return

    count = len(results)
    label = "result" if count == 1 else "results"
    print(f'\nFound {count} refined semantic {label} for: "{query}"\n')
    print("=" * 75)

    if group_by_doc:
        doc_groups = defaultdict(list)
        for r in results:
            doc_groups[r.chunk.filename].append(r)

        for doc_name, items in doc_groups.items():
            print(f"\n[DOCUMENT] {doc_name} ({len(items)} matching passages)")
            print("-" * 75)
            for idx, res in enumerate(items, start=1):
                chunk = res.chunk
                snippet = chunk.text.replace("\n", " ")
                snippet = re.sub(r"\s+", " ", snippet)
                snippet = snippet.encode("ascii", errors="replace").decode("ascii")

                print(f"  {idx}. Page: {chunk.page_number}/{chunk.total_pages} | Score: {res.score:.4f} | ID: {chunk.chunk_id}")
                print(f'     "{snippet}"\n')
    else:
        for i, res in enumerate(results, start=1):
            chunk = res.chunk
            snippet = chunk.text.replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet)
            snippet = snippet.encode("ascii", errors="replace").decode("ascii")

            print(f"\n{i}. Document: {chunk.filename}")
            print(f"   Hierarchy: Document -> Page {chunk.page_number}/{chunk.total_pages} -> Chunk #{chunk.chunk_index}")
            print(f"   Score: {res.score:.4f} | Chunk ID: {chunk.chunk_id}")
            if show_metadata and chunk.metadata.get("title"):
                print(f"   Doc Title: {chunk.metadata.get('title')}")
            print(f'   "{snippet}"')
            print("-" * 75)
