"""
Tests for LLM Engine (V4 — LLM Backends & Extractive Synthesizer).
"""

from unittest.mock import patch, MagicMock
import pytest

from src.llm_engine import (
    BaseLLM,
    ExtractiveFallbackLLM,
    OllamaLLM,
    LlamaCppLLM,
    get_llm_engine,
)


class TestExtractiveFallbackLLM:

    def test_synthesize_grounded_answer(self):
        synthesizer = ExtractiveFallbackLLM()
        prompt = (
            "System prompt...\n\n"
            "--- CONTEXT PASSAGES ---\n"
            "[1] Document: crypto.pdf (Page 63)\n"
            "RSA uses prime numbers p and q. The product n = pq is difficult to factor.\n\n"
            "[2] Document: bio.pdf (Page 2)\n"
            "Photosynthesis is a completely unrelated biological process.\n"
            "--- END CONTEXT ---\n\n"
            "User Question: What prime numbers are used in RSA?\n\n"
            "Grounded Answer (citing [1], [2], etc.):"
        )

        answer = synthesizer.generate(prompt)
        assert len(answer) > 0
        assert "[1]" in answer
        assert "RSA" in answer or "prime" in answer

    def test_empty_context_handling(self):
        synthesizer = ExtractiveFallbackLLM()
        prompt = (
            "System prompt...\n\n"
            "--- CONTEXT PASSAGES ---\n"
            "No relevant passages found.\n"
            "--- END CONTEXT ---\n\n"
            "User Question: What is gravity?\n\n"
            "Grounded Answer:"
        )

        answer = synthesizer.generate(prompt)
        assert "do not contain enough information" in answer.lower()


class TestLLMFactory:

    def test_get_extractive_engine(self):
        engine = get_llm_engine("extractive")
        assert isinstance(engine, ExtractiveFallbackLLM)

    def test_get_ollama_engine(self):
        engine = get_llm_engine("ollama", model_name_or_path="mistral")
        assert isinstance(engine, OllamaLLM)
        assert engine.model == "mistral"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            get_llm_engine("nonexistent_backend")

    def test_llama_cpp_requires_path(self):
        with pytest.raises(ValueError, match="model_path is required"):
            get_llm_engine("llama_cpp", model_name_or_path=None)
