"""
Tests for ResultPostprocessor (V3 — Deduplication, Reranking, and Filtering).
"""

import pytest

from src.chunker import TextChunk
from src.vector_store import ScoredChunk
from src.config import RetrievalConfig
from src.postprocessing import ResultPostprocessor, compute_token_jaccard


class TestResultPostprocessor:

    def test_compute_token_jaccard(self):
        t1 = "attention mechanism in neural networks"
        t2 = "the attention mechanism for neural network models"
        jaccard = compute_token_jaccard(t1, t2)
        assert jaccard > 0.3

        # Empty strings
        assert compute_token_jaccard("", "") == 0.0
        assert compute_token_jaccard("word", "") == 0.0

    def test_threshold_filtering(self):
        config = RetrievalConfig(score_threshold=0.5)
        postprocessor = ResultPostprocessor(config)

        results = [
            ScoredChunk(TextChunk("c1", "a.pdf", "/a.pdf", 1, "text high", chunk_index=0), score=0.8),
            ScoredChunk(TextChunk("c2", "a.pdf", "/a.pdf", 1, "text low", chunk_index=1), score=0.4),
        ]

        processed = postprocessor.postprocess(results)
        assert len(processed) == 1
        assert processed[0].chunk.chunk_id == "c1"

    def test_near_duplicate_suppression(self):
        config = RetrievalConfig(deduplicate=True, score_threshold=0.0)
        postprocessor = ResultPostprocessor(config)

        chunk1 = TextChunk("c1", "a.pdf", "/a.pdf", 1, "The quick brown fox jumps over the lazy dog", chunk_index=0)
        chunk2 = TextChunk("c2", "a.pdf", "/a.pdf", 1, "The quick brown fox jumps over the lazy dog repeatedly", chunk_index=5)  # Far index but identical text

        results = [
            ScoredChunk(chunk1, score=0.9),
            ScoredChunk(chunk2, score=0.85),
        ]

        processed = postprocessor.postprocess(results, jaccard_threshold=0.7)
        # Should drop chunk2 because it's nearly identical to chunk1
        assert len(processed) == 1
        assert processed[0].chunk.chunk_id == "c1"

    def test_same_page_adjacent_chunk_suppression(self):
        config = RetrievalConfig(deduplicate=True, min_chunk_distance=1, score_threshold=0.0)
        postprocessor = ResultPostprocessor(config)

        # Immediately adjacent chunks (chunk_index 0 and 0 are distance 0 < 1)
        chunk1 = TextChunk("c1", "a.pdf", "/a.pdf", 1, "First unique part of text", chunk_index=0)
        chunk2 = TextChunk("c2", "a.pdf", "/a.pdf", 1, "Second unique part of text", chunk_index=0)

        results = [
            ScoredChunk(chunk1, score=0.9),
            ScoredChunk(chunk2, score=0.85),
        ]

        processed = postprocessor.postprocess(results)
        assert len(processed) == 1
        assert processed[0].chunk.chunk_id == "c1"

    def test_document_diversity_capping(self):
        config = RetrievalConfig(score_threshold=0.0, deduplicate=False)
        postprocessor = ResultPostprocessor(config)

        results = [
            ScoredChunk(TextChunk("c1", "a.pdf", "/a.pdf", 1, "text 1"), score=0.9),
            ScoredChunk(TextChunk("c2", "a.pdf", "/a.pdf", 2, "text 2"), score=0.8),
            ScoredChunk(TextChunk("c3", "b.pdf", "/b.pdf", 1, "text 3"), score=0.7),
        ]

        # Limit to 1 chunk per document
        processed = postprocessor.postprocess(results, max_chunks_per_doc=1)
        assert len(processed) == 2
        doc_names = [r.chunk.filename for r in processed]
        assert doc_names == ["a.pdf", "b.pdf"]
