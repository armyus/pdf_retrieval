"""
Vector Store — V2: In-Memory Vector Store

Stores chunks and their dense embeddings, computing cosine similarities
efficiently using NumPy.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.chunker import TextChunk


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its similarity score."""
    chunk: TextChunk
    score: float


class InMemoryVectorStore:
    """Lightweight in-memory vector index for storing and querying text chunk embeddings."""

    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape (N, D)

    @property
    def count(self) -> int:
        return len(self.chunks)

    def add(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Adds chunks and corresponding embeddings to the store.

        Args:
            chunks: List of TextChunk objects.
            embeddings: np.ndarray of shape (len(chunks), D).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})")

        self.chunks.extend(chunks)
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0) -> List[ScoredChunk]:
        """
        Searches for the most similar chunks to the query vector.
        Assumes normalized vectors (cosine similarity = dot product).

        Args:
            query_vector: np.ndarray of shape (D,).
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score cutoff.

        Returns:
            List of ScoredChunk sorted in descending order of similarity score.
        """
        if self.embeddings is None or len(self.chunks) == 0 or top_k <= 0:
            return []

        # Ensure 1D query vector
        query_vector = np.squeeze(query_vector).astype(np.float32)

        # Compute dot products (cosine similarity for normalized vectors)
        scores = np.dot(self.embeddings, query_vector)

        # Filter by threshold
        valid_indices = np.where(scores >= score_threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Get top-k indices
        valid_scores = scores[valid_indices]
        if len(valid_scores) > top_k:
            # Use argpartition for fast top_k selection
            top_partition = np.argpartition(valid_scores, -top_k)[-top_k:]
            sorted_top_indices = top_partition[np.argsort(-valid_scores[top_partition])]
            selected_indices = valid_indices[sorted_top_indices]
        else:
            sorted_order = np.argsort(-valid_scores)
            selected_indices = valid_indices[sorted_order]

        results = [
            ScoredChunk(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
            )
            for idx in selected_indices
        ]

        return results

    def clear(self) -> None:
        """Clears all stored chunks and vectors."""
        self.chunks = []
        self.embeddings = None
