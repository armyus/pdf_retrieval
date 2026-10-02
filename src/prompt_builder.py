"""
Prompt Builder — V4: Context Construction for RAG

Constructs structured prompts for LLM generation with clear citation
guidelines, grounding constraints, and source numbering.
"""

from typing import List, Tuple
from src.vector_store import ScoredChunk


class PromptBuilder:
    """Formats retrieved document chunks into grounded RAG prompts."""

    SYSTEM_PROMPT = (
        "You are an AI assistant grounded strictly in the provided document passages. "
        "Answer the question using ONLY the facts directly stated in the context passages below. "
        "Do NOT invent, assume, or extrapolate facts beyond the text. "
        "Cite the sources of your facts using bracketed numbers (e.g. [1], [2]). "
        "If the passages do not contain sufficient evidence or information to answer the question, "
        "clearly state: 'The provided documents do not contain enough information to answer this question.'"
    )

    @classmethod
    def format_context(cls, chunks: List[ScoredChunk]) -> Tuple[str, List[dict]]:
        """
        Formats retrieved ScoredChunks into a numbered reference context block.

        Returns:
            Tuple of (formatted_context_string, source_metadata_list)
        """
        if not chunks:
            return "No relevant passages found.", []

        context_lines = []
        sources = []

        for idx, scored in enumerate(chunks, start=1):
            chunk = scored.chunk
            clean_text = chunk.text.strip().replace("\r\n", " ").replace("\n", " ")
            context_lines.append(
                f"[{idx}] Document: {chunk.filename} (Page {chunk.page_number})\n{clean_text}\n"
            )
            sources.append({
                "index": idx,
                "filename": chunk.filename,
                "full_path": chunk.full_path,
                "page_number": chunk.page_number,
                "total_pages": chunk.total_pages,
                "chunk_id": chunk.chunk_id,
                "score": scored.score,
                "snippet": clean_text[:200] + ("..." if len(clean_text) > 200 else ""),
            })

        return "\n".join(context_lines), sources

    @classmethod
    def build_prompt(cls, query: str, chunks: List[ScoredChunk]) -> str:
        """
        Constructs the complete prompt string including system instructions, context, and query.
        """
        context_str, _ = cls.format_context(chunks)

        prompt = (
            f"{cls.SYSTEM_PROMPT}\n\n"
            f"--- CONTEXT PASSAGES ---\n"
            f"{context_str}\n"
            f"--- END CONTEXT ---\n\n"
            f"User Question: {query}\n\n"
            f"Grounded Answer (citing [1], [2], etc.):"
        )
        return prompt
