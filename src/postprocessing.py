"""
Postprocessing — V3: Deduplication, Reranking, and Filtering

Applies near-duplicate suppression, score threshold filtering,
source diversity capping, and result re-ranking.
"""

from typing import List, Optional, Set
import re

from src.vector_store import ScoredChunk
from src.config import RetrievalConfig


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Computes word-level Jaccard similarity between two text snippets."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


class ResultPostprocessor:
    """Cleans, deduplicates, and filters retrieved search results."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig.default()

    def postprocess(
        self,
        results: List[ScoredChunk],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        deduplicate: Optional[bool] = None,
        max_chunks_per_doc: Optional[int] = None,
        jaccard_threshold: float = 0.65,
    ) -> List[ScoredChunk]:
        """
        Applies filtering, deduplication, and diversity capping.

        Args:
            results: Raw scored chunk results sorted by score descending.
            top_k: Number of final results desired.
            score_threshold: Minimum similarity threshold.
            deduplicate: Whether to suppress near duplicates.
            max_chunks_per_doc: Maximum chunks permitted per document.
            jaccard_threshold: Threshold above which two chunks are considered duplicates.

        Returns:
            Filtered and deduplicated list of ScoredChunk objects.
        """
        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        do_dedup = deduplicate if deduplicate is not None else self.config.deduplicate
        doc_cap = max_chunks_per_doc if max_chunks_per_doc is not None else self.config.max_chunks_per_document

        # 1. Filter by score threshold
        filtered = [r for r in results if r.score >= threshold]

        if not filtered:
            return []

        # 2. Near-duplicate suppression & same-page adjacency suppression
        if do_dedup:
            deduped: List[ScoredChunk] = []
            seen_page_chunks: Set[str] = set()

            for item in filtered:
                chunk = item.chunk
                page_key = f"{chunk.filename}_p{chunk.page_number}"

                # Check if we already kept an immediately adjacent chunk on the same page
                is_adjacent = False
                for existing in deduped:
                    if (
                        existing.chunk.filename == chunk.filename
                        and existing.chunk.page_number == chunk.page_number
                        and abs(existing.chunk.chunk_index - chunk.chunk_index) < self.config.min_chunk_distance
                    ):
                        is_adjacent = True
                        break

                if is_adjacent:
                    continue

                # Check text overlap (Jaccard similarity) against already accepted results
                is_duplicate_text = False
                for existing in deduped:
                    sim = compute_token_jaccard(existing.chunk.text, chunk.text)
                    if sim >= jaccard_threshold:
                        is_duplicate_text = True
                        break

                if not is_duplicate_text:
                    deduped.append(item)

            filtered = deduped

        # 3. Document diversity capping (if enabled)
        if doc_cap is not None and doc_cap > 0:
            capped: List[ScoredChunk] = []
            doc_counts: dict = {}
            for item in filtered:
                doc_name = item.chunk.filename
                count = doc_counts.get(doc_name, 0)
                if count < doc_cap:
                    capped.append(item)
                    doc_counts[doc_name] = count + 1
            filtered = capped

        # 4. Truncate to top_k
        return filtered[:k]
