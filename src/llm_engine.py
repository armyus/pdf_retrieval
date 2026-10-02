"""
LLM Engine — V4: Local LLM Interface & Backends

Supports multiple local LLM execution backends:
- Ollama (Local REST API)
- Llama-cpp (Local GGUF models)
- HuggingFace Transformers (Local pipeline)
- Extractive/Synthesizer Fallback (Zero-dependency local answer construction)
"""

import abc
import json
import logging
import urllib.request
import urllib.error
import re
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLM(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text completion for the given prompt."""
        pass


class OllamaLLM(BaseLLM):
    """Generates answers via local Ollama instance (http://localhost:11434)."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`). Details: {e}"
            )


class LlamaCppLLM(BaseLLM):
    """Generates answers via llama-cpp-python and a local GGUF file."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def _load_model(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is not installed. Install with `pip install llama-cpp-python`."
                )

    def generate(self, prompt: str) -> str:
        self._load_model()
        output = self._llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["--- END CONTEXT ---", "User Question:"],
        )
        return output["choices"][0]["text"].strip()


class ExtractiveFallbackLLM(BaseLLM):
    """
    Lightweight deterministic synthesizer when no heavy local LLM server is active.
    Extracts the most relevant grounded statements directly from the context passages
    and attaches correct citation numbers.
    """

    def generate(self, prompt: str) -> str:
        # Check if context was empty
        if "No relevant passages found." in prompt:
            return "The provided documents do not contain enough information to answer this question."

        # Extract passages section
        context_match = re.search(r"--- CONTEXT PASSAGES ---\s*(.*?)\s*--- END CONTEXT ---", prompt, re.DOTALL)
        query_match = re.search(r"User Question:\s*(.*?)\n\nGrounded Answer", prompt, re.DOTALL)

        if not context_match or not query_match:
            return "The provided documents do not contain enough information to answer this question."

        context_text = context_match.group(1).strip()
        query = query_match.group(1).strip()

        # Parse numbered passages
        passages = re.findall(r"\[(\d+)\]\s*Document:\s*([^\n]+)\s*\(Page\s*(\d+)\)\s*\n(.*?)(?=\[\d+\]|\Z)", context_text, re.DOTALL)

        if not passages:
            return "The provided documents do not contain enough information to answer this question."

        # Query keywords
        query_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 2)

        answers = []
        for idx, doc_name, page_num, body in passages:
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", body) if len(s.strip()) > 20]
            # Score sentences by query word overlap
            scored_sentences = []
            for s in sentences:
                s_words = set(w.lower() for w in re.findall(r"\w+", s))
                overlap = len(query_words.intersection(s_words))
                if overlap > 0:
                    scored_sentences.append((overlap, s))

            scored_sentences.sort(key=lambda x: -x[0])
            if scored_sentences:
                best_s = scored_sentences[0][1]
                answers.append(f"{best_s} [{idx}]")

        if not answers:
            # Fall back to using first sentence of top passage
            first_body = passages[0][3].strip()
            first_s = re.split(r"(?<=[.?!])\s+", first_body)[0]
            return f"{first_s} [{passages[0][0]}]"

        return " ".join(answers[:3])


def get_llm_engine(backend: str = "auto", model_name_or_path: Optional[str] = None) -> BaseLLM:
    """
    Factory function to instantiate the appropriate LLM backend.

    Args:
        backend: 'ollama', 'llama_cpp', 'extractive', or 'auto'
        model_name_or_path: Model name for Ollama or file path for llama-cpp.
    """
    backend_lower = backend.lower()

    if backend_lower == "ollama":
        model = model_name_or_path or "llama3.2"
        return OllamaLLM(model=model)

    elif backend_lower in ("llama_cpp", "llamacpp", "gguf"):
        if not model_name_or_path:
            raise ValueError("model_path is required for llama_cpp backend.")
        return LlamaCppLLM(model_path=model_name_or_path)

    elif backend_lower == "extractive":
        return ExtractiveFallbackLLM()

    elif backend_lower == "auto":
        # Try Ollama first (check if accessible)
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0):
                return OllamaLLM(model=model_name_or_path or "llama3.2")
        except Exception:
            pass

        # Fallback to extractive synthesizer
        return ExtractiveFallbackLLM()

    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Choose from: 'ollama', 'llama_cpp', 'extractive', 'auto'.")
