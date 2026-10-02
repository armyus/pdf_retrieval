"""
RAG Pipeline — V4: Retrieval-Augmented Generation

Orchestrates semantic retrieval, context prompt construction,
local LLM answer synthesis, and source citation mapping.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.retriever import SemanticRetriever
from src.prompt_builder import PromptBuilder
from src.llm_engine import BaseLLM, get_llm_engine


@dataclass
class RAGResult:
    """Represents a complete grounded answer with citations and sources."""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    has_sufficient_context: bool = True
    raw_prompt: Optional[str] = None


class RAGPipeline:
    """End-to-end RAG question answering pipeline."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        llm: Optional[BaseLLM] = None,
        llm_backend: str = "auto",
        model_name_or_path: Optional[str] = None,
    ):
        self.retriever = retriever
        self.llm = llm or get_llm_engine(backend=llm_backend, model_name_or_path=model_name_or_path)

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        deduplicate: bool = True,
    ) -> RAGResult:
        """
        Retrieves relevant passages and generates a grounded answer with citations.

        Args:
            query: User question.
            top_k: Number of context passages to retrieve.
            score_threshold: Minimum similarity threshold.
            deduplicate: Whether to deduplicate retrieved chunks.

        Returns:
            RAGResult containing the answer, sources, and context status.
        """
        # 1. Retrieve relevant chunks
        chunks = self.retriever.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            deduplicate=deduplicate,
        )

        # 2. Check if any context was found
        if not chunks:
            return RAGResult(
                query=query,
                answer="The provided documents do not contain enough information to answer this question.",
                sources=[],
                has_sufficient_context=False,
            )

        # 3. Construct grounded prompt & extract source metadata
        prompt = PromptBuilder.build_prompt(query, chunks)
        _, sources = PromptBuilder.format_context(chunks)

        # 4. Generate answer from LLM
        raw_answer = self.llm.generate(prompt)

        # Check if LLM indicates insufficient information
        insufficient_phrases = [
            "do not contain enough information",
            "not enough information",
            "cannot be answered from the provided",
            "no relevant passages found",
        ]
        has_sufficient = not any(phrase in raw_answer.lower() for phrase in insufficient_phrases)

        return RAGResult(
            query=query,
            answer=raw_answer,
            sources=sources,
            has_sufficient_context=has_sufficient,
            raw_prompt=prompt,
        )


def _safe_console_str(text: str) -> str:
    """Sanitizes strings for safe printing on Windows cp1252 consoles."""
    return text.encode("ascii", errors="replace").decode("ascii")


def print_rag_result(result: RAGResult) -> None:
    """Pretty-prints a RAG answer with citation details."""
    print("\n" + "=" * 75)
    print(f"Question: {_safe_console_str(result.query)}")
    print("=" * 75)

    print("\nAnswer:")
    print(_safe_console_str(result.answer))

    if result.sources and result.has_sufficient_context:
        print("\nSources & Citations:")
        for s in result.sources:
            print(f"  [{s['index']}] {s['filename']} — Page {s['page_number']}/{s['total_pages']} (Score: {s['score']:.4f})")
            print(f"      Excerpt: \"{_safe_console_str(s['snippet'])}\"\n")
    elif not result.has_sufficient_context:
        print("\nNote: Insufficient evidence in the provided documents to form a grounded response.")

    print("-" * 75)
