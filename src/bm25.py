"""
BM25 Engine — V5: Pure Python BM25Okapi Implementation

Provides tokenization, inverted index, IDF computation, and BM25Okapi scoring
for sparse lexical search across document text chunks.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set

from src.chunker import TextChunk
from src.vector_store import ScoredChunk


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercased alphanumeric terms."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25Okapi index for lexical chunk retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (standard: 1.2 - 2.0).
            b: Document length normalization parameter (standard: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> number of docs containing term
        self.term_freqs: List[Counter] = []                # doc_idx -> Counter(term -> count)
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def index(self, chunks: List[TextChunk]) -> None:
        """Indexes a list of TextChunk objects."""
        self.chunks = list(chunks)
        self.total_docs = len(self.chunks)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}

        if self.total_docs == 0:
            self.avg_doc_len = 0.0
            return

        total_len = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / self.total_docs if self.total_docs > 0 else 0.0

        # Precompute Lucene/Okapi standard IDF: ln(1 + (N - n + 0.5) / (n + 0.5))
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """
        Scores all indexed chunks against the query using BM25Okapi.

        Returns:
            List of ScoredChunk sorted by score descending.
        """
        if self.total_docs == 0 or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.total_docs

        for term in query_tokens:
            if term not in self.idf:
                continue

            term_idf = self.idf[term]

            for doc_idx in range(self.total_docs):
                tf = self.term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += term_idf * (numerator / denominator)

        # Pair scores with chunks and filter out 0.0 scores
        results = [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in range(self.total_docs)
            if scores[i] > 0.0
        ]

        results.sort(key=lambda x: -x.score)
        return results[:top_k]
