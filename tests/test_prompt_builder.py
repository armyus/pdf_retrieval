"""
Tests for PromptBuilder (V4 — Prompt Construction for RAG).
"""

from src.chunker import TextChunk
from src.vector_store import ScoredChunk
from src.prompt_builder import PromptBuilder


class TestPromptBuilder:

    def test_format_empty_context(self):
        context_str, sources = PromptBuilder.format_context([])
        assert "No relevant passages found" in context_str
        assert sources == []

    def test_format_context_with_chunks(self):
        chunks = [
            ScoredChunk(
                chunk=TextChunk("c1", "paper1.pdf", "/p1.pdf", 3, "Attention is all you need for transformers.", total_pages=10),
                score=0.88,
            ),
            ScoredChunk(
                chunk=TextChunk("c2", "paper2.pdf", "/p2.pdf", 7, "Self-attention computes dynamic weights.", total_pages=15),
                score=0.75,
            ),
        ]

        context_str, sources = PromptBuilder.format_context(chunks)

        assert "[1] Document: paper1.pdf (Page 3)" in context_str
        assert "[2] Document: paper2.pdf (Page 7)" in context_str
        assert "Attention is all you need" in context_str

        assert len(sources) == 2
        assert sources[0]["index"] == 1
        assert sources[0]["filename"] == "paper1.pdf"
        assert sources[0]["page_number"] == 3
        assert sources[0]["score"] == 0.88
        assert sources[1]["index"] == 2

    def test_build_prompt(self):
        chunks = [
            ScoredChunk(
                chunk=TextChunk("c1", "crypto.pdf", "/c.pdf", 1, "RSA uses large prime numbers."),
                score=0.9,
            )
        ]
        prompt = PromptBuilder.build_prompt("How does RSA work?", chunks)

        assert "--- CONTEXT PASSAGES ---" in prompt
        assert "User Question: How does RSA work?" in prompt
        assert "Grounded Answer" in prompt
        assert "RSA uses large prime numbers." in prompt
