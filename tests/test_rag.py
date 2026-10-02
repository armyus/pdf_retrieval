"""
Tests for RAGPipeline (V4 — Grounded RAG Pipeline).
"""

from unittest.mock import MagicMock
import pytest

from src.rag import RAGPipeline, RAGResult, print_rag_result
from src.chunker import TextChunk
from src.vector_store import ScoredChunk
from src.retriever import SemanticRetriever
from src.llm_engine import BaseLLM


class DummyLLM(BaseLLM):
    def __init__(self, response: str = "This is a grounded answer from docs [1]."):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class TestRAGPipeline:

    def test_answer_with_retrieved_context(self):
        retriever = MagicMock(spec=SemanticRetriever)
        retriever.search.return_value = [
            ScoredChunk(
                chunk=TextChunk("c1", "crypto.pdf", "/c.pdf", 66, "RSA uses large primes to ensure secrecy of messages."),
                score=0.85,
            )
        ]

        llm = DummyLLM(response="RSA relies on large primes for confidentiality [1].")
        pipeline = RAGPipeline(retriever=retriever, llm=llm)

        result = pipeline.answer("How does RSA work?")

        assert isinstance(result, RAGResult)
        assert result.has_sufficient_context is True
        assert "RSA relies on large primes" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0]["filename"] == "crypto.pdf"
        assert result.sources[0]["page_number"] == 66

    def test_answer_with_no_retrieved_context(self):
        retriever = MagicMock(spec=SemanticRetriever)
        retriever.search.return_value = []

        llm = DummyLLM()
        pipeline = RAGPipeline(retriever=retriever, llm=llm)

        result = pipeline.answer("What is string theory?")

        assert result.has_sufficient_context is False
        assert "do not contain enough information" in result.answer.lower()
        assert result.sources == []

    def test_answer_with_insufficient_llm_response(self):
        retriever = MagicMock(spec=SemanticRetriever)
        retriever.search.return_value = [
            ScoredChunk(chunk=TextChunk("c1", "a.pdf", "/a.pdf", 1, "Unrelated text"), score=0.4)
        ]

        llm = DummyLLM(response="The provided documents do not contain enough information to answer this question.")
        pipeline = RAGPipeline(retriever=retriever, llm=llm)

        result = pipeline.answer("Complex query?")
        assert result.has_sufficient_context is False

    def test_print_rag_result(self, capsys):
        result = RAGResult(
            query="Test query",
            answer="Grounded answer citing [1].",
            sources=[{
                "index": 1,
                "filename": "doc.pdf",
                "page_number": 2,
                "total_pages": 10,
                "score": 0.92,
                "snippet": "Snippet text",
            }],
            has_sufficient_context=True,
        )

        print_rag_result(result)
        out = capsys.readouterr().out

        assert "Question: Test query" in out
        assert "Grounded answer citing [1]." in out
        assert "[1] doc.pdf — Page 2/10" in out
