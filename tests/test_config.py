"""
Tests for RetrievalConfig (V5 — Configuration Presets & Strategies).
"""

from src.config import RetrievalConfig


class TestRetrievalConfig:

    def test_default_config(self):
        config = RetrievalConfig.default()
        assert config.strategy == "hybrid"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 75
        assert config.top_k == 5
        assert config.fusion_method == "rrf"
        assert config.deduplicate is True

    def test_dense_only_preset(self):
        config = RetrievalConfig.dense_only()
        assert config.strategy == "dense"
        assert config.score_threshold == 0.30

    def test_bm25_only_preset(self):
        config = RetrievalConfig.bm25_only()
        assert config.strategy == "bm25"

    def test_precise_preset(self):
        config = RetrievalConfig.precise()
        assert config.chunk_size == 350
        assert config.chunk_overlap == 50
        assert config.top_k == 5
        assert config.dense_weight == 0.7

    def test_broad_preset(self):
        config = RetrievalConfig.broad()
        assert config.chunk_size == 800
        assert config.chunk_overlap == 120
        assert config.top_k == 8
