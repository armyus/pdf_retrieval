"""
Retriever — V3: Enhanced Semantic Retrieval Pipeline

Coordinates chunking, embedding generation, vector indexing,
metadata filtering, deduplication, and hierarchical result presentation.
"""

from typing import List, Optional, Dict, Any, Callable
import re
from collections import defaultdict

from src.pdf_parser import ParsedPDF
from src.chunker import TextChunk, TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import InMemoryVectorStore, ScoredChunk
from src.config import RetrievalConfig
from src.postprocessing import ResultPostprocessor


class SemanticRetriever:
    """End-to-end semantic retriever with filtering and deduplication for PDF documents."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = config or RetrievalConfig.default()

        # Allow explicit parameter overrides
        c_size = chunk_size if chunk_size is not None else self.config.chunk_size
        c_overlap = chunk_overlap if chunk_overlap is not None else self.config.chunk_overlap
        emb_model = embedding_model if embedding_model is not None else self.config.embedding_model
        dev = device if device is not None else self.config.device

        self.chunker = TextChunker(chunk_size=c_size, chunk_overlap=c_overlap)
        self.embedder = EmbeddingGenerator(model_name=emb_model, device=dev)
        self.vector_store = InMemoryVectorStore()
        self.postprocessor = ResultPostprocessor(self.config)
        self._indexed = False

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def total_chunks(self) -> int:
        return self.vector_store.count

    def index_documents(self, documents: List[ParsedPDF], batch_size: Optional[int] = None) -> int:
        """
        Chunks and indexes parsed PDF documents into the vector store.

        Args:
            documents: List of ParsedPDF objects.
            batch_size: Embedding batch size.

        Returns:
            Number of indexed text chunks.
        """
        self.vector_store.clear()
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            self._indexed = True
            return 0

        bs = batch_size if batch_size is not None else self.config.batch_size
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts, batch_size=bs, normalize=True)
        self.vector_store.add(chunks, embeddings)
        self._indexed = True
        return len(chunks)

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
        Executes semantic search with deduplication and optional document filtering.

        Args:
            query: Natural language query string.
            top_k: Number of most similar chunks to return.
            score_threshold: Minimum cosine similarity score.
            deduplicate: Whether to remove near duplicates and adjacent page chunks.
            max_chunks_per_doc: Cap results from a single document.
            doc_filter: Optional filter predicate accepting filename (returns True to include).

        Returns:
            List of ScoredChunk objects sorted by similarity score descending.
        """
        if not query or not query.strip() or not self._indexed or self.vector_store.count == 0:
            return []

        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold

        # Request more candidates from vector store to allow postprocessor filtering
        candidate_k = max(k * 4, 20)
        query_vector = self.embedder.embed_query(query.strip(), normalize=True)

        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            score_threshold=threshold,
        )

        # Apply metadata document filtering if provided
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


def print_semantic_results(
    results: List[ScoredChunk],
    query: str,
    show_metadata: bool = True,
    group_by_doc: bool = False,
) -> None:
    """Pretty-print semantic retrieval search results with hierarchical metadata."""
    if not results:
        print(f'\nNo semantic results found matching the criteria for: "{query}"')
        return

    count = len(results)
    label = "result" if count == 1 else "results"
    print(f'\nFound {count} refined semantic {label} for: "{query}"\n')
    print("=" * 75)

    if group_by_doc:
        # Group by document
        doc_groups = defaultdict(list)
        for r in results:
            doc_groups[r.chunk.filename].append(r)

        for doc_name, items in doc_groups.items():
            print(f"\n[DOCUMENT] {doc_name} ({len(items)} matching passages)")
            print("-" * 75)
            for idx, res in enumerate(items, start=1):
                chunk = res.chunk
                snippet = chunk.text.replace("\n", " ")
                snippet = re.sub(r"\s+", " ", snippet)
                snippet = snippet.encode("ascii", errors="replace").decode("ascii")

                print(f"  {idx}. Page: {chunk.page_number}/{chunk.total_pages} | Score: {res.score:.4f} | ID: {chunk.chunk_id}")
                print(f'     "{snippet}"\n')
    else:
        for i, res in enumerate(results, start=1):
            chunk = res.chunk
            snippet = chunk.text.replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet)
            snippet = snippet.encode("ascii", errors="replace").decode("ascii")

            print(f"\n{i}. Document: {chunk.filename}")
            print(f"   Hierarchy: Document -> Page {chunk.page_number}/{chunk.total_pages} -> Chunk #{chunk.chunk_index}")
            print(f"   Similarity Score: {res.score:.4f} | Chunk ID: {chunk.chunk_id}")
            if show_metadata and chunk.metadata.get("title"):
                print(f"   Doc Title: {chunk.metadata.get('title')}")
            print(f'   "{snippet}"')
            print("-" * 75)
