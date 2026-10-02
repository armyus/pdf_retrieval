"""
Evaluation Framework — V6: Information Retrieval Metrics & Benchmark Suite

Provides quantitative evaluation metrics (Precision@K, Recall@K, MRR, MAP, nDCG@K)
and automated comparative benchmarking across retrieval strategies (BM25, Dense, Hybrid).
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from src.retrievers import BaseRetriever, get_retriever
from src.pdf_parser import ParsedPDF
from src.config import RetrievalConfig


@dataclass
class EvaluationQuery:
    """Represents a benchmark query with expected ground-truth relevant documents."""
    query_id: str
    query: str
    relevant_docs: List[str]  # Substrings or filenames of relevant documents
    relevant_keywords: List[str] = field(default_factory=list)  # Keywords expected in matching chunks


@dataclass
class QueryMetrics:
    """Metrics computed for a single evaluation query."""
    query_id: str
    query: str
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float
    average_precision: float
    ndcg_at_k: Dict[int, float]
    retrieved_count: int
    relevant_retrieved_count: int


@dataclass
class BenchmarkResult:
    """Aggregated evaluation metrics for a retrieval strategy."""
    strategy: str
    total_queries: int
    mean_precision_at_k: Dict[int, float]
    mean_recall_at_k: Dict[int, float]
    mean_reciprocal_rank: float  # MRR
    mean_average_precision: float  # MAP
    mean_ndcg_at_k: Dict[int, float]
    query_results: List[QueryMetrics] = field(default_factory=list)


def is_chunk_relevant(doc_filename: str, chunk_text: str, eval_query: EvaluationQuery) -> bool:
    """Determines whether a retrieved chunk matches ground truth expectations."""
    # Check document filename match
    doc_match = any(rel.lower() in doc_filename.lower() for rel in eval_query.relevant_docs)
    if not doc_match:
        return False

    # If specific keywords are specified, check for presence
    if eval_query.relevant_keywords:
        kw_match = any(kw.lower() in chunk_text.lower() for kw in eval_query.relevant_keywords)
        return kw_match

    return True


def compute_precision_at_k(relevance_binary: List[bool], k: int) -> float:
    """Computes Precision@K: (number of relevant items in top k) / k."""
    if k <= 0:
        return 0.0
    top_k_rel = relevance_binary[:k]
    return sum(1 for r in top_k_rel if r) / k


def compute_recall_at_k(relevance_binary: List[bool], k: int, total_relevant: int) -> float:
    """Computes Recall@K: (number of relevant items in top k) / (total relevant items)."""
    if total_relevant <= 0 or k <= 0:
        return 0.0
    top_k_rel = relevance_binary[:k]
    return min(1.0, sum(1 for r in top_k_rel if r) / total_relevant)


def compute_reciprocal_rank(relevance_binary: List[bool]) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant item (or 0.0)."""
    for rank, rel in enumerate(relevance_binary, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def compute_average_precision(relevance_binary: List[bool], total_relevant: int) -> float:
    """Computes Average Precision (AP)."""
    if total_relevant <= 0:
        return 0.0

    running_sum = 0.0
    rel_count = 0
    for rank, rel in enumerate(relevance_binary, start=1):
        if rel:
            rel_count += 1
            running_sum += rel_count / rank

    return running_sum / total_relevant if total_relevant > 0 else 0.0


def compute_ndcg_at_k(relevance_binary: List[bool], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K)."""
    if k <= 0 or not relevance_binary:
        return 0.0

    top_k = relevance_binary[:k]
    # DCG = sum( rel_i / log2(i + 1) )
    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(top_k))

    # IDCG: Best possible DCG with all 1s first
    ideal_k = sorted(top_k, reverse=True)
    idcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(ideal_k))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class Evaluator:
    """Automated retrieval evaluation and benchmark suite."""

    # Standard default benchmark queries for technical PDF collection
    DEFAULT_TEST_QUERIES = [
        EvaluationQuery(
            query_id="q1",
            query="How does RSA encryption use prime numbers for cryptographic security?",
            relevant_docs=["Cryptography and Number theory", "crypto"],
            relevant_keywords=["RSA", "prime", "modulus", "factor"],
        ),
        EvaluationQuery(
            query_id="q2",
            query="What are the key graphical techniques and visual methods in exploratory data analysis?",
            relevant_docs=["EXPLORATORY DATA ANALYSIS", "EDA"],
            relevant_keywords=["graphical", "univariate", "scatter", "histogram", "multidimensional"],
        ),
        EvaluationQuery(
            query_id="q3",
            query="How do artificial intelligence and human computer interaction collaborate in spatial and interface design?",
            relevant_docs=["Artificial Intelligence in HCI", "HCI"],
            relevant_keywords=["HCI", "interface", "interaction", "design"],
        ),
        EvaluationQuery(
            query_id="q4",
            query="Modular arithmetic algorithms and divisibility properties in number theory",
            relevant_docs=["Cryptography and Number theory", "crypto"],
            relevant_keywords=["modular", "congruence", "divisor", "algorithm"],
        ),
        EvaluationQuery(
            query_id="q5",
            query="Handling missing data, outliers, and data pollution in data preparation",
            relevant_docs=["EXPLORATORY DATA ANALYSIS", "EDA"],
            relevant_keywords=["missing", "outlier", "pollution", "noise"],
        ),
    ]

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values or [1, 3, 5]

    def evaluate_query(
        self,
        retriever: BaseRetriever,
        eval_query: EvaluationQuery,
    ) -> QueryMetrics:
        """Evaluates a single query on the given retriever."""
        max_k = max(self.k_values)
        results = retriever.search(eval_query.query, top_k=max_k, deduplicate=True)

        # Build binary relevance vector
        relevance_binary = [
            is_chunk_relevant(r.chunk.filename, r.chunk.text, eval_query)
            for r in results
        ]

        # Estimated total relevant items (at least 1 for non-empty ground truth)
        total_relevant = max(1, len(eval_query.relevant_docs))

        p_at_k = {k: compute_precision_at_k(relevance_binary, k) for k in self.k_values}
        r_at_k = {k: compute_recall_at_k(relevance_binary, k, total_relevant) for k in self.k_values}
        rr = compute_reciprocal_rank(relevance_binary)
        ap = compute_average_precision(relevance_binary, total_relevant)
        ndcg = {k: compute_ndcg_at_k(relevance_binary, k) for k in self.k_values}

        return QueryMetrics(
            query_id=eval_query.query_id,
            query=eval_query.query,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=rr,
            average_precision=ap,
            ndcg_at_k=ndcg,
            retrieved_count=len(results),
            relevant_retrieved_count=sum(1 for r in relevance_binary if r),
        )

    def evaluate_retriever(
        self,
        retriever: BaseRetriever,
        queries: Optional[List[EvaluationQuery]] = None,
        strategy_name: str = "retriever",
    ) -> BenchmarkResult:
        """Evaluates a retriever over a list of test queries and aggregates metrics."""
        test_queries = queries or self.DEFAULT_TEST_QUERIES
        query_metrics_list = [self.evaluate_query(retriever, q) for q in test_queries]

        num_queries = len(test_queries)
        if num_queries == 0:
            return BenchmarkResult(
                strategy=strategy_name,
                total_queries=0,
                mean_precision_at_k={},
                mean_recall_at_k={},
                mean_reciprocal_rank=0.0,
                mean_average_precision=0.0,
                mean_ndcg_at_k={},
            )

        mean_p_k = {
            k: sum(m.precision_at_k.get(k, 0.0) for m in query_metrics_list) / num_queries
            for k in self.k_values
        }
        mean_r_k = {
            k: sum(m.recall_at_k.get(k, 0.0) for m in query_metrics_list) / num_queries
            for k in self.k_values
        }
        mean_rr = sum(m.reciprocal_rank for m in query_metrics_list) / num_queries
        mean_ap = sum(m.average_precision for m in query_metrics_list) / num_queries
        mean_ndcg = {
            k: sum(m.ndcg_at_k.get(k, 0.0) for m in query_metrics_list) / num_queries
            for k in self.k_values
        }

        return BenchmarkResult(
            strategy=strategy_name,
            total_queries=num_queries,
            mean_precision_at_k=mean_p_k,
            mean_recall_at_k=mean_r_k,
            mean_reciprocal_rank=mean_rr,
            mean_average_precision=mean_ap,
            mean_ndcg_at_k=mean_ndcg,
            query_results=query_metrics_list,
        )

    def run_comparative_benchmark(
        self,
        documents: List[ParsedPDF],
        strategies: Optional[List[str]] = None,
        queries: Optional[List[EvaluationQuery]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Runs side-by-side benchmark comparing multiple retrieval strategies."""
        strats = strategies or ["bm25", "dense", "hybrid"]
        test_queries = queries or self.DEFAULT_TEST_QUERIES
        results: Dict[str, BenchmarkResult] = {}

        for strat in strats:
            config = RetrievalConfig.default()
            config.strategy = strat
            retriever = get_retriever(strat, config)
            retriever.index_documents(documents)
            bench = self.evaluate_retriever(retriever, test_queries, strategy_name=strat)
            results[strat] = bench

        return results


def print_benchmark_summary(results: Dict[str, BenchmarkResult]) -> None:
    """Pretty-prints a comparative benchmark leaderboard table."""
    print("\n" + "=" * 80)
    print("  RETRIEVAL BENCHMARK LEADERBOARD (V6 Evaluation)")
    print("=" * 80)

    header = f"{'Strategy':<12} | {'P@1':<8} | {'P@3':<8} | {'P@5':<8} | {'MRR':<8} | {'MAP':<8} | {'nDCG@5':<8}"
    print(header)
    print("-" * 80)

    for strat, res in results.items():
        p1 = res.mean_precision_at_k.get(1, 0.0)
        p3 = res.mean_precision_at_k.get(3, 0.0)
        p5 = res.mean_precision_at_k.get(5, 0.0)
        mrr = res.mean_reciprocal_rank
        map_s = res.mean_average_precision
        ndcg5 = res.mean_ndcg_at_k.get(5, 0.0)

        line = f"{strat.upper():<12} | {p1:<8.4f} | {p3:<8.4f} | {p5:<8.4f} | {mrr:<8.4f} | {map_s:<8.4f} | {ndcg5:<8.4f}"
        print(line)

    print("=" * 80 + "\n")
