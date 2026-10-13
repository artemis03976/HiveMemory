from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hivememory.config.patchouli import RerankerConfig
from hivememory.config.shared import EmbeddingConfig
from hivememory.infrastructure.embedding.base import SingletonModelService
from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService
from hivememory.infrastructure.rerank.fast_embed_reranker import FastEmbedRerankerService


class TestBGEM3EmbeddingService:
    def test_load_model_registers_custom_xenova_model(self):
        config = EmbeddingConfig(model_name="Xenova/bge-m3")
        service = BGEM3EmbeddingService(config=config)
        service._model = None

        mock_text_embedding_cls = MagicMock()
        mock_model_source = MagicMock()
        mock_pooling_type = MagicMock()
        mock_pooling_type.CLS = "CLS"

        with (
            patch("fastembed.TextEmbedding", mock_text_embedding_cls),
            patch("fastembed.common.model_description.ModelSource", mock_model_source),
            patch("fastembed.common.model_description.PoolingType", mock_pooling_type),
        ):
            service._load_model()

        mock_text_embedding_cls.add_custom_model.assert_called_once()
        kwargs = mock_text_embedding_cls.add_custom_model.call_args.kwargs
        assert kwargs["model"] == "Xenova/bge-m3"
        assert kwargs["model_file"] == "onnx/model_int8.onnx"
        assert kwargs["dim"] == 1024
        assert kwargs["normalization"] is True
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

        with (
            patch("fastembed.TextEmbedding", mock_text_embedding_cls),
            patch("fastembed.common.model_description.ModelSource", mock_model_source),
            patch("fastembed.common.model_description.PoolingType", mock_pooling_type),
        ):
            service._load_model()

        mock_text_embedding_cls.assert_called_once_with(
            model_name="Xenova/bge-m3",
            cache_dir=config.cache_dir,
        )


class _RecordingDenseModel:
    """按文本长度生成确定向量，并记录每次 embed 的输入。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self._fail = fail

    def embed(self, texts):
        self.calls.append(list(texts))
        if self._fail:
            raise RuntimeError("onnx failure")
        return [np.array([float(len(text)), 0.5]) for text in texts]


@pytest.fixture
def make_encode_service():
    """以独立模型名创建实例并注入假模型；结束时移出 multiton 实例表。"""
    keys: list[str] = []

    def _make(model: _RecordingDenseModel) -> BGEM3EmbeddingService:
        config = EmbeddingConfig(model_name="test/bge-m3-encode")
        service = BGEM3EmbeddingService(config=config)
        service._model = model
        keys.append(f"{config.model_name}:{config.device}")
        return service

    yield _make
    for key in keys:
        SingletonModelService._instances.pop(key, None)


class TestBGEM3Encode:
    """BGE-M3 encode 按参数组合返回的形状（存储层依赖 dense 与 dense+sparse 两种）。"""

    def test_encode_single_dense_text_returns_one_vector(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        assert service.encode(dense_texts="abc") == pytest.approx([3.0, 0.5])

    def test_encode_dense_batch_returns_vector_per_text_in_order(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        result = service.encode(dense_texts=["a", "bb"])

        assert len(result) == 2
        assert result[0] == pytest.approx([1.0, 0.5])
        assert result[1] == pytest.approx([2.0, 0.5])

    def test_encode_dense_and_sparse_returns_vector_with_raw_sparse_text(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        result = service.encode(dense_texts="abc", sparse_texts="abc raw")

        assert set(result) == {"dense", "sparse_text"}
        assert result["dense"] == pytest.approx([3.0, 0.5])
        assert result["sparse_text"] == "abc raw"

    def test_encode_sparse_only_returns_raw_text_without_dense_model(self, make_encode_service):
        model = _RecordingDenseModel()
        service = make_encode_service(model)

        assert service.encode(sparse_texts="raw") == "raw"
        assert model.calls == []

    def test_encode_without_any_text_raises_value_error(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        with pytest.raises(ValueError, match="至少需要提供"):
            service.encode()

    @pytest.mark.parametrize(
        ("dense_texts", "expected"),
        [("abc", []), (["a", "bb"], [[], []])],
        ids=["single", "batch"],
    )
    def test_encode_degrades_to_empty_vectors_when_model_fails(
        self, make_encode_service, dense_texts, expected
    ):
        service = make_encode_service(_RecordingDenseModel(fail=True))

        assert service.encode(dense_texts=dense_texts) == expected


class TestFastEmbedRerankerService:
    def test_load_model_uses_configured_cache_dir(self):
        config = RerankerConfig(
            model_name="BAAI/bge-reranker-base",
            cache_dir="data/model_cache",
        )
        service = FastEmbedRerankerService(config=config)
        service._model = None

        mock_cross_encoder_cls = MagicMock()

        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", mock_cross_encoder_cls):
            service._load_model()

        mock_cross_encoder_cls.assert_called_once_with(
            model_name="BAAI/bge-reranker-base",
            cache_dir="data/model_cache",
        )

    def test_compute_score_coerces_numpy_like_scores_to_float(self):
        config = RerankerConfig(model_name="BAAI/bge-reranker-base")
        service = FastEmbedRerankerService(config=config)

        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            MagicMock(__float__=lambda _self: 0.6),
            MagicMock(__float__=lambda _self: -0.1),
        ]
        service._model = mock_model

        scores = service.compute_score(
            [
                ["query", "doc-a"],
                ["query", "doc-b"],
            ]
        )

        assert scores == [0.6, -0.1]
