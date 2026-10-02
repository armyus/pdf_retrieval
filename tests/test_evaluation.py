"""
Tests for Evaluation Framework (V6 — Evaluation Metrics & Benchmarking).
"""

import pytest

from src.evaluation import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_average_precision,
    compute_ndcg_at_k,
    is_chunk_relevant,
    EvaluationQuery,
    Evaluator,
)
from src.retrievers import BM25Retriever
from src.pdf_parser import PageContent, ParsedPDF


class TestEvaluationMetrics:

    def test_precision_at_k(self):
        # 3 items, 2 relevant
        relevance = [True, False, True, False, False]
        assert compute_precision_at_k(relevance, 1) == 1.0
        assert compute_precision_at_k(relevance, 2) == 0.5
        assert compute_precision_at_k(relevance, 3) == pytest.approx(2 / 3)
        assert compute_precision_at_k([], 5) == 0.0

    def test_recall_at_k(self):
        relevance = [True, False, True]
        total_relevant = 2
        assert compute_recall_at_k(relevance, 1, total_relevant) == 0.5
        assert compute_recall_at_k(relevance, 2, total_relevant) == 0.5
        assert compute_recall_at_k(relevance, 3, total_relevant) == 1.0
        assert compute_recall_at_k(relevance, 3, 0) == 0.0

    def test_reciprocal_rank(self):
        assert compute_reciprocal_rank([True, False, False]) == 1.0
        assert compute_reciprocal_rank([False, True, False]) == 0.5
        assert compute_reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
        assert compute_reciprocal_rank([False, False, False]) == 0.0

    def test_average_precision(self):
        # [Rel, NonRel, Rel], total 2 rel
        # P@1 = 1/1 = 1.0, P@2 = 1/2 = 0.5, P@3 = 2/3 = 0.666
        # AP = (1.0 + 2/3) / 2 = 1.666/2 = 0.8333
        relevance = [True, False, True]
        ap = compute_average_precision(relevance, total_relevant=2)
        assert ap == pytest.approx((1.0 + (2 / 3)) / 2)

    def test_ndcg_at_k(self):
        # Perfect ranking: [True, True, False]
        perfect = [True, True, False]
        assert compute_ndcg_at_k(perfect, 3) == 1.0

        # Inverted ranking: [False, True, True]
        imperfect = [False, True, True]
        ndcg = compute_ndcg_at_k(imperfect, 3)
        assert 0.0 < ndcg < 1.0

        # Zero relevance
        assert compute_ndcg_at_k([False, False], 2) == 0.0

    def test_is_chunk_relevant(self):
        query = EvaluationQuery(
            query_id="q1",
            query="test query",
            relevant_docs=["crypto.pdf"],
            relevant_keywords=["RSA", "prime"],
        )

        # Matching doc and keyword
        assert is_chunk_relevant("crypto.pdf", "Discussion of RSA encryption.", query) is True
        # Wrong doc
        assert is_chunk_relevant("biology.pdf", "Discussion of RSA encryption.", query) is False
        # Right doc, wrong keyword
        assert is_chunk_relevant("crypto.pdf", "General history of ciphers.", query) is False


class TestEvaluatorPipeline:

    def test_evaluate_retriever(self):
        doc = ParsedPDF(
            filename="crypto.pdf",
            full_path="/crypto.pdf",
            pages=[PageContent(1, "RSA uses large prime numbers for encryption.")],
            total_pages=1,
        )

        retriever = BM25Retriever()
        retriever.index_documents([doc])

        query = EvaluationQuery(
            query_id="q1",
            query="RSA prime encryption",
            relevant_docs=["crypto.pdf"],
        )

        evaluator = Evaluator(k_values=[1, 3])
        bench = evaluator.evaluate_retriever(retriever, queries=[query])

        assert bench.total_queries == 1
        assert bench.mean_reciprocal_rank == 1.0
        assert bench.mean_precision_at_k[1] == 1.0
