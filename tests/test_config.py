"""
Tests for RetrievalConfig (V3 — Configuration Module).
"""

from src.config import RetrievalConfig


class TestRetrievalConfig:

    def test_default_config(self):
        config = RetrievalConfig.default()
        assert config.chunk_size == 500
        assert config.chunk_overlap == 75
        assert config.top_k == 5
        assert config.score_threshold == 0.30
        assert config.deduplicate is True

    def test_precise_preset(self):
        config = RetrievalConfig.precise()
        assert config.chunk_size == 350
        assert config.chunk_overlap == 50
        assert config.score_threshold == 0.40
        assert config.top_k == 5

    def test_broad_preset(self):
        config = RetrievalConfig.broad()
        assert config.chunk_size == 800
        assert config.chunk_overlap == 120
        assert config.score_threshold == 0.25
        assert config.top_k == 8
