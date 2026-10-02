"""
Swappable Retrievers — V5: Modular Retrieval Architecture

Provides unified BaseRetriever interface and implementations for:
  - DenseRetriever (Vector embeddings)
  - BM25Retriever (Sparse lexical search)
  - HybridRetriever (Reciprocal Rank Fusion & Weighted Score Fusion)
  - KeywordRetriever (Exact substring matching)
"""

import abc
from typing import List, Optional, Dict, Callable, Any
from collections import defaultdict

from src.pdf_parser import ParsedPDF
from src.chunker import TextChunk, TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import InMemoryVectorStore, ScoredChunk
from src.bm25 import BM25Index
from src.keyword_search import KeywordSearcher
from src.config import RetrievalConfig
from src.postprocessing import ResultPostprocessor


class BaseRetriever(abc.ABC):
    """Abstract base class for all retrieval strategies."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig.default()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.postprocessor = ResultPostprocessor(self.config)
        self._indexed = False
        self._total_chunks = 0

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    @abc.abstractmethod
    def index_documents(self, documents: List[ParsedPDF]) -> int:
        """Indexes parsed PDF documents. Returns number of chunks indexed."""
        pass

    @abc.abstractmethod
    def _raw_search(self, query: str, candidate_k: int) -> List[ScoredChunk]:
        """Executes core retrieval algorithm returning candidate scored chunks."""
        pass

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        deduplicate: Optional[bool] = None,
        max_chunks_per_doc: Optional[int] = None,
        doc_filter: Optional[Callable[[str], bool]] = None,
    ) -> List[ScoredChunk]:
        """
        Public search method with candidate over-fetching, document filtering,
        deduplication, and diversity capping.
        """
        if not query or not query.strip() or not self._indexed or self._total_chunks == 0:
            return []

        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold

        # Request more candidates to give postprocessor room for deduplication
        candidate_k = max(k * 4, 25)
        raw_results = self._raw_search(query.strip(), candidate_k)

        # Apply metadata document filter
        if doc_filter is not None:
            raw_results = [r for r in raw_results if doc_filter(r.chunk.filename)]

        # Apply postprocessing (deduplication, diversity capping, truncation to top_k)
        final_results = self.postprocessor.postprocess(
            results=raw_results,
            top_k=k,
            score_threshold=threshold,
            deduplicate=deduplicate,
            max_chunks_per_doc=max_chunks_per_doc,
        )

        return final_results


class DenseRetriever(BaseRetriever):
    """Dense vector embedding retriever using sentence-transformers and InMemoryVectorStore."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self.embedder = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            device=self.config.device,
        )
        self.vector_store = InMemoryVectorStore()

    def index_documents(self, documents: List[ParsedPDF]) -> int:
        self.vector_store.clear()
        chunks = self.chunker.chunk_documents(documents)
        self._total_chunks = len(chunks)

        if not chunks:
            self._indexed = True
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(
            texts,
            batch_size=self.config.batch_size,
            normalize=True,
        )
        self.vector_store.add(chunks, embeddings)
        self._indexed = True
        return self._total_chunks

    def _raw_search(self, query: str, candidate_k: int) -> List[ScoredChunk]:
        query_vector = self.embedder.embed_query(query, normalize=True)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            score_threshold=0.0,
        )


class BM25Retriever(BaseRetriever):
    """Sparse lexical retriever using BM25Okapi."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self.bm25_index = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)

    def index_documents(self, documents: List[ParsedPDF]) -> int:
        chunks = self.chunker.chunk_documents(documents)
        self._total_chunks = len(chunks)
        self.bm25_index.index(chunks)
        self._indexed = True
        return self._total_chunks

    def _raw_search(self, query: str, candidate_k: int) -> List[ScoredChunk]:
        return self.bm25_index.search(query, top_k=candidate_k)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever fusing dense vector search and sparse BM25 lexical search.
    Supports Reciprocal Rank Fusion (RRF) and Weighted Score Normalization.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self.dense_retriever = DenseRetriever(config=self.config)
        self.bm25_retriever = BM25Retriever(config=self.config)

    def index_documents(self, documents: List[ParsedPDF]) -> int:
        dense_count = self.dense_retriever.index_documents(documents)
        bm25_count = self.bm25_retriever.index_documents(documents)
        self._total_chunks = dense_count
        self._indexed = True
        return self._total_chunks

    def _raw_search(self, query: str, candidate_k: int) -> List[ScoredChunk]:
        # Fetch candidate rankings from both retrievers
        dense_results = self.dense_retriever._raw_search(query, candidate_k)
        bm25_results = self.bm25_retriever._raw_search(query, candidate_k)

        if self.config.fusion_method == "weighted":
            return self._weighted_score_fusion(dense_results, bm25_results, candidate_k)
        else:
            return self._reciprocal_rank_fusion(dense_results, bm25_results, candidate_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[ScoredChunk],
        bm25_results: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        """
        Combines rankings using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
        """
        rrf_k = self.config.rrf_k
        w_dense = self.config.dense_weight
        w_sparse = self.config.sparse_weight

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, TextChunk] = {}

        # Accumulate dense ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item.chunk.chunk_id
            chunk_map[cid] = item.chunk
            rrf_scores[cid] += w_dense / (rrf_k + rank)

        # Accumulate BM25 ranks
        for rank, item in enumerate(bm25_results, start=1):
            cid = item.chunk.chunk_id
            chunk_map[cid] = item.chunk
            rrf_scores[cid] += w_sparse / (rrf_k + rank)

        # Convert to ScoredChunk list sorted by fused score
        fused = [
            ScoredChunk(chunk=chunk_map[cid], score=score)
            for cid, score in rrf_scores.items()
        ]
        fused.sort(key=lambda x: -x.score)
        return fused[:top_k]

    def _weighted_score_fusion(
        self,
        dense_results: List[ScoredChunk],
        bm25_results: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        """Combines normalized scores via weighted linear combination."""
        w_dense = self.config.dense_weight
        w_sparse = self.config.sparse_weight

        # Min-max normalize dense scores
        dense_norm = self._min_max_normalize(dense_results)
        # Min-max normalize BM25 scores
        bm25_norm = self._min_max_normalize(bm25_results)

        combined_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, TextChunk] = {}

        for item, norm_score in dense_norm:
            cid = item.chunk.chunk_id
            chunk_map[cid] = item.chunk
            combined_scores[cid] += w_dense * norm_score

        for item, norm_score in bm25_norm:
            cid = item.chunk.chunk_id
            chunk_map[cid] = item.chunk
            combined_scores[cid] += w_sparse * norm_score

        fused = [
            ScoredChunk(chunk=chunk_map[cid], score=score)
            for cid, score in combined_scores.items()
        ]
        fused.sort(key=lambda x: -x.score)
        return fused[:top_k]

    @staticmethod
    def _min_max_normalize(results: List[ScoredChunk]) -> List[tuple]:
        if not results:
            return []
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [(r, 1.0) for r in results]
        return [(r, (r.score - min_s) / (max_s - min_s)) for r in results]


class KeywordRetriever(BaseRetriever):
    """Retriever using exact case-insensitive keyword match counts."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self.searcher = KeywordSearcher()
        self.chunks: List[TextChunk] = []

    def index_documents(self, documents: List[ParsedPDF]) -> int:
        self.chunks = self.chunker.chunk_documents(documents)
        self._total_chunks = len(self.chunks)
        self._indexed = True
        return self._total_chunks

    def _raw_search(self, query: str, candidate_k: int) -> List[ScoredChunk]:
        import re
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
        scored = []
        for chunk in self.chunks:
            matches = len(pattern.findall(chunk.text))
            if matches > 0:
                scored.append(ScoredChunk(chunk=chunk, score=float(matches)))

        scored.sort(key=lambda x: -x.score)
        return scored[:candidate_k]


def get_retriever(strategy: str = "hybrid", config: Optional[RetrievalConfig] = None) -> BaseRetriever:
    """
    Factory function for swappable retrievers.

    Args:
        strategy: 'hybrid', 'dense', 'bm25', 'keyword'
        config: RetrievalConfig instance
    """
    cfg = config or RetrievalConfig.default()
    strat_lower = strategy.lower()

    if strat_lower == "hybrid":
        return HybridRetriever(config=cfg)
    elif strat_lower in ("dense", "semantic"):
        return DenseRetriever(config=cfg)
    elif strat_lower == "bm25":
        return BM25Retriever(config=cfg)
    elif strat_lower == "keyword":
        return KeywordRetriever(config=cfg)
    else:
        raise ValueError(
            f"Unknown retrieval strategy: '{strategy}'. Choose from: 'hybrid', 'dense', 'bm25', 'keyword'."
        )
