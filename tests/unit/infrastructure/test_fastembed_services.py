import pytest
from unittest.mock import MagicMock, patch

from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService
from hivememory.infrastructure.rerank.fast_embed_reranker import FastEmbedRerankerService
from hivememory.system.config import EmbeddingConfig, RerankerConfig


class TestBGEM3EmbeddingService:
    def test_load_model_registers_custom_xenova_model(self):
        config = EmbeddingConfig(model_name="Xenova/bge-m3")
        service = BGEM3EmbeddingService(config=config)
        service._model = None

        mock_text_embedding_cls = MagicMock()
        mock_model_source = MagicMock()
        mock_pooling_type = MagicMock()
        mock_pooling_type.CLS = "CLS"

        with patch("fastembed.TextEmbedding", mock_text_embedding_cls), \
             patch("fastembed.common.model_description.ModelSource", mock_model_source), \
             patch("fastembed.common.model_description.PoolingType", mock_pooling_type):
            service._load_model()

        mock_text_embedding_cls.add_custom_model.assert_called_once()
        kwargs = mock_text_embedding_cls.add_custom_model.call_args.kwargs
        assert kwargs["model"] == "Xenova/bge-m3"
        assert kwargs["model_file"] == "onnx/model_int8.onnx"
        assert kwargs["dim"] == 1024
        assert kwargs["normalization"] is True
        assert kwargs["pooling"] == "CLS"
        mock_text_embedding_cls.assert_called_once_with(
            model_name="Xenova/bge-m3",
            cache_dir=config.cache_dir,
        )

    def test_load_model_ignores_duplicate_custom_model_registration(self):
        config = EmbeddingConfig(model_name="Xenova/bge-m3")
        service = BGEM3EmbeddingService(config=config)
        service._model = None

        mock_text_embedding_cls = MagicMock()
        mock_text_embedding_cls.add_custom_model.side_effect = ValueError("already exists")
        mock_model_source = MagicMock()
        mock_pooling_type = MagicMock()
        mock_pooling_type.CLS = "CLS"

        with patch("fastembed.TextEmbedding", mock_text_embedding_cls), \
             patch("fastembed.common.model_description.ModelSource", mock_model_source), \
             patch("fastembed.common.model_description.PoolingType", mock_pooling_type):
            service._load_model()

        mock_text_embedding_cls.assert_called_once_with(
            model_name="Xenova/bge-m3",
            cache_dir=config.cache_dir,
        )


class TestFastEmbedRerankerService:
    def test_load_model_uses_configured_cache_dir(self):
        config = RerankerConfig(
            model_name="BAAI/bge-reranker-base",
            cache_dir="data/model_cache",
        )
        service = FastEmbedRerankerService(config=config)
        service._model = None
        service.cache_dir = config.cache_dir

        mock_cross_encoder_cls = MagicMock()

        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", mock_cross_encoder_cls):
            service._load_model()

        mock_cross_encoder_cls.assert_called_once_with(
            model_name="BAAI/bge-reranker-base",
            cache_dir="data/model_cache",
        )

    def test_compute_score_returns_float_scores_in_input_order(self):
        config = RerankerConfig(model_name="BAAI/bge-reranker-base")
        service = FastEmbedRerankerService(config=config)

        mock_model = MagicMock()
        mock_model.rerank.return_value = [0.2, 0.8]
        service._model = mock_model

        scores = service.compute_score([
            ["query", "doc-a"],
            ["query", "doc-b"],
        ])

        assert scores == [0.2, 0.8]
        mock_model.rerank.assert_called_once_with("query", ["doc-a", "doc-b"], batch_size=256)

    def test_compute_score_coerces_numpy_like_scores_to_float(self):
        config = RerankerConfig(model_name="BAAI/bge-reranker-base")
        service = FastEmbedRerankerService(config=config)

        mock_model = MagicMock()
        mock_model.rerank.return_value = [MagicMock(__float__=lambda _self: 0.6), MagicMock(__float__=lambda _self: -0.1)]
        service._model = mock_model

        scores = service.compute_score([
            ["query", "doc-a"],
            ["query", "doc-b"],
        ])

        assert scores == [0.6, -0.1]
