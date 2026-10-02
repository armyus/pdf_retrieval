"""
Embeddings — V2: Local Embedding Generator

Wraps sentence-transformers to generate dense vector embeddings
for text chunks and search queries.
"""

from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """Generates dense vector embeddings using sentence-transformers."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Args:
            model_name: Name of the HuggingFace / sentence-transformers model.
            device: Device to run embeddings on ('cpu', 'cuda', etc.).
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load model to avoid overhead until first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Generates embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            normalize: Whether to L2-normalize vectors for cosine similarity.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generates embedding for a single query string.

        Args:
            query: Natural language query string.
            normalize: Whether to L2-normalize vector.

        Returns:
            np.ndarray of shape (embedding_dim,) with dtype float32.
        """
        self._load_model()
        embedding = self._model.encode(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
