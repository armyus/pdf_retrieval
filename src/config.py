"""
Configuration — V3: System and Retrieval Configuration

Provides typed configuration presets and settings for chunking,
embedding models, similarity filtering, deduplication, and ranking.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """Configuration settings for PDF indexing, chunking, and retrieval."""

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 75

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32

    # Retrieval settings
    top_k: int = 5
    score_threshold: float = 0.30

    # Post-processing / Deduplication settings
    deduplicate: bool = True
    min_chunk_distance: int = 1  # Suppress immediately adjacent overlapping chunks if from same page
    max_chunks_per_document: Optional[int] = None  # Cap chunks per doc to encourage source diversity

    @classmethod
    def default(cls) -> "RetrievalConfig":
        """Default balanced retrieval configuration."""
        return cls()

    @classmethod
    def precise(cls) -> "RetrievalConfig":
        """Higher precision configuration with smaller chunks and higher threshold."""
        return cls(
            chunk_size=350,
            chunk_overlap=50,
            score_threshold=0.40,
            top_k=5,
        )

    @classmethod
    def broad(cls) -> "RetrievalConfig":
        """Broader context configuration with larger chunks and lower threshold."""
        return cls(
            chunk_size=800,
            chunk_overlap=120,
            score_threshold=0.25,
            top_k=8,
        )
