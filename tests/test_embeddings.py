"""
Tests for EmbeddingGenerator (V2 — Local Embedding Generator).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:

    def test_init_defaults(self):
        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.device == "cpu"
        assert generator._model is None

    def test_empty_texts_returns_empty_array(self):
        generator = EmbeddingGenerator()
        res = generator.embed_texts([])
        assert isinstance(res, np.ndarray)
        assert res.shape == (0, 0)

    def test_embed_texts_mocked(self):
        generator = EmbeddingGenerator()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        generator._model = mock_model

        res = generator.embed_texts(["hello", "world"])
        assert res.shape == (2, 2)
        mock_model.encode.assert_called_once()

    def test_embed_query_mocked(self):
        generator = EmbeddingGenerator()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5, 0.6], dtype=np.float32)
        generator._model = mock_model

        res = generator.embed_query("query")
        assert res.shape == (2,)
        mock_model.encode.assert_called_once()
