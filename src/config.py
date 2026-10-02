"""
Configuration — V5: System, Hybrid, and Retrieval Configuration

Provides typed configuration presets and settings for chunking,
embedding models, BM25 parameters, fusion strategies, and ranking.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """Configuration settings for PDF indexing, chunking, and retrieval."""

    # Retrieval strategy: 'hybrid', 'dense', 'bm25', 'keyword'
    strategy: str = "hybrid"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 75

    # Embedding settings (Dense)
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32

    # BM25 settings (Sparse)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid Fusion settings
    fusion_method: str = "rrf"        # 'rrf' (Reciprocal Rank Fusion) or 'weighted'
    rrf_k: int = 60                   # Standard RRF constant
    dense_weight: float = 0.6         # Weight for dense vector scores in weighted fusion
    sparse_weight: float = 0.4        # Weight for BM25 scores in weighted fusion

    # General Retrieval settings
    top_k: int = 5
    score_threshold: float = 0.0      # 0.0 default for hybrid/rrf, tuned per preset

    # Post-processing / Deduplication settings
    deduplicate: bool = True
    min_chunk_distance: int = 1       # Suppress immediately adjacent overlapping chunks if from same page
    max_chunks_per_document: Optional[int] = None  # Cap chunks per doc to encourage source diversity

    @classmethod
    def default(cls) -> "RetrievalConfig":
        """Default balanced hybrid retrieval configuration."""
        return cls()

    @classmethod
    def dense_only(cls) -> "RetrievalConfig":
        """Dense embedding retrieval only."""
        return cls(strategy="dense", score_threshold=0.30)

    @classmethod
    def bm25_only(cls) -> "RetrievalConfig":
        """BM25 sparse lexical retrieval only."""
        return cls(strategy="bm25", score_threshold=0.0)

    @classmethod
    def precise(cls) -> "RetrievalConfig":
        """Higher precision hybrid configuration with smaller chunks."""
        return cls(
            strategy="hybrid",
            chunk_size=350,
            chunk_overlap=50,
            score_threshold=0.0,
            top_k=5,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

    @classmethod
    def broad(cls) -> "RetrievalConfig":
        """Broader context configuration with larger chunks."""
        return cls(
            strategy="hybrid",
            chunk_size=800,
            chunk_overlap=120,
            score_threshold=0.0,
            top_k=8,
            dense_weight=0.5,
            sparse_weight=0.5,
        )
