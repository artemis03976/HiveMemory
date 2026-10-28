import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from hivememory.config.patchouli import RerankerConfig
from hivememory.config.shared import EmbeddingConfig
from hivememory.infrastructure.embedding import bge_m3
from hivememory.infrastructure.embedding.base import SingletonModelService
from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService, get_bge_m3_service
from hivememory.infrastructure.rerank.fast_embed_reranker import FastEmbedRerankerService


def _word_tokenizer() -> Tokenizer:
    """按空白切词的真实分词器，初始截断与 bge-m3 自带的 8192 token 一致。"""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=8192)
    return tokenizer


@pytest.fixture
def text_embedding_cls(monkeypatch):
    """替换 fastembed 的 TextEmbedding；构造出的模型带一个真实分词器。"""
    cls = MagicMock()
    cls.return_value.model.tokenizer = _word_tokenizer()
    monkeypatch.setattr("fastembed.TextEmbedding", cls)
    return cls


@pytest.fixture
def unloaded_bge_m3(monkeypatch):
    """进程级 multiton 实例；测试结束时恢复它原来的模型。"""
    service = BGEM3EmbeddingService(config=EmbeddingConfig(model_name="Xenova/bge-m3"))
    monkeypatch.setattr(service, "_model", None)
    return service


class TestBGEM3EmbeddingService:
    def test_load_model_registers_custom_xenova_model_without_cpu_memory_arena(
        self, text_embedding_cls, unloaded_bge_m3
    ):
        unloaded_bge_m3._load_model()

        kwargs = text_embedding_cls.add_custom_model.call_args.kwargs
        assert kwargs["model"] == "Xenova/bge-m3"
        assert kwargs["model_file"] == "onnx/model_int8.onnx"
        assert kwargs["dim"] == 1024
        assert kwargs["normalization"] is True
        text_embedding_cls.assert_called_once_with(
            model_name="Xenova/bge-m3",
            cache_dir=unloaded_bge_m3.cache_dir,
            enable_cpu_mem_arena=False,
        )

    def test_load_model_ignores_duplicate_custom_model_registration(
        self, text_embedding_cls, unloaded_bge_m3
    ):
        text_embedding_cls.add_custom_model.side_effect = ValueError("already exists")

        unloaded_bge_m3._load_model()

        assert unloaded_bge_m3.is_loaded() is True

    def test_loaded_model_truncates_input_to_1024_tokens(self, text_embedding_cls, unloaded_bge_m3):
        """输入按 1024 token 截断，而不是模型自带的 8192 token。"""
        unloaded_bge_m3._load_model()

        tokenizer = text_embedding_cls.return_value.model.tokenizer
        assert len(tokenizer.encode(" ".join(["word"] * 5000)).ids) == 1024

    def test_load_model_without_tokenizer_fails_and_stays_unloaded(
        self, text_embedding_cls, unloaded_bge_m3
    ):
        """无法设置截断时加载失败，不交出未截断的模型。"""
        text_embedding_cls.return_value.model = object()

        with pytest.raises(RuntimeError, match="分词器"):
            unloaded_bge_m3._load_model()

        assert unloaded_bge_m3.is_loaded() is False

    def test_get_service_uses_bge_m3_when_configured_model_is_not_bge_m3(self, monkeypatch):
        """存储层只支持 BGE-M3：配置为其他模型时改用 Xenova/bge-m3。"""
        monkeypatch.setattr(bge_m3, "_bge_m3_instance", None)

        service = get_bge_m3_service(
            EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        assert service.model_name == "Xenova/bge-m3"


class _RecordingDenseModel:
    """按文本长度生成确定向量，并记录每次 embed 的输入与所在线程。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.threads: list[int] = []
        self._fail = fail

    def embed(self, texts):
        self.calls.append(list(texts))
        self.threads.append(threading.get_ident())
        if self._fail:
            raise RuntimeError("onnx failure")
        return [np.array([float(len(text)), 0.5]) for text in texts]


@pytest_asyncio.fixture
async def make_encode_service():
    """以独立模型名创建实例并注入假模型；结束时释放推理线程并移出 multiton 实例表。"""
    services: list[BGEM3EmbeddingService] = []

    def _make(model: _RecordingDenseModel) -> BGEM3EmbeddingService:
        config = EmbeddingConfig(model_name="test/bge-m3-encode")
        service = BGEM3EmbeddingService(config=config)
        service._model = model
        services.append(service)
        return service

    yield _make
    for service in services:
        await service.close()
        SingletonModelService._instances.pop(service.model_name, None)


class TestBGEM3Encode:
    """BGE-M3 encode 按参数组合返回的形状（存储层依赖 dense 与 dense+sparse 两种）。"""

    @pytest.mark.asyncio
    async def test_encode_single_dense_text_returns_one_vector(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        assert await service.encode(dense_texts="abc") == pytest.approx([3.0, 0.5])

    @pytest.mark.asyncio
    async def test_encode_dense_batch_returns_vector_per_text_in_order(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        result = await service.encode(dense_texts=["a", "bb"])

        assert len(result) == 2
        assert result[0] == pytest.approx([1.0, 0.5])
        assert result[1] == pytest.approx([2.0, 0.5])

    @pytest.mark.asyncio
    async def test_encode_dense_and_sparse_returns_vector_with_raw_sparse_text(
        self, make_encode_service
    ):
        service = make_encode_service(_RecordingDenseModel())

        result = await service.encode(dense_texts="abc", sparse_texts="abc raw")

        assert set(result) == {"dense", "sparse_text"}
        assert result["dense"] == pytest.approx([3.0, 0.5])
        assert result["sparse_text"] == "abc raw"

    @pytest.mark.asyncio
    async def test_encode_sparse_only_returns_raw_text_without_dense_model(
        self, make_encode_service
    ):
        model = _RecordingDenseModel()
        service = make_encode_service(model)

        assert await service.encode(sparse_texts="raw") == "raw"
        assert model.calls == []

    @pytest.mark.asyncio
    async def test_encode_without_any_text_raises_value_error(self, make_encode_service):
        service = make_encode_service(_RecordingDenseModel())

        with pytest.raises(ValueError, match="至少需要提供"):
            await service.encode()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("dense_texts", "expected"),
        [("abc", []), (["a", "bb"], [[], []])],
        ids=["single", "batch"],
    )
    async def test_encode_degrades_to_empty_vectors_when_model_fails(
        self, make_encode_service, dense_texts, expected
    ):
        service = make_encode_service(_RecordingDenseModel(fail=True))

        assert await service.encode(dense_texts=dense_texts) == expected

    @pytest.mark.asyncio
    async def test_encode_loads_and_embeds_off_the_event_loop_thread(
        self, make_encode_service, monkeypatch
    ):
        """首次编码时的模型加载与稠密编码都在推理线程中执行。"""
        model = _RecordingDenseModel()
        service = make_encode_service(model)
        load_threads: list[int] = []

        def load_model() -> None:
            load_threads.append(threading.get_ident())
            service._model = model

        monkeypatch.setattr(service, "_model", None)
        monkeypatch.setattr(service, "_load_model", load_model)

        await service.encode(dense_texts="abc")

        loop_thread = threading.get_ident()
        assert len(load_threads) == 1
        assert load_threads[0] != loop_thread
        assert model.threads == load_threads


@pytest_asyncio.fixture
async def reranker_service():
    """重排服务是进程级单例；测试结束时释放它的推理线程。"""
    service = FastEmbedRerankerService(config=RerankerConfig(model_name="BAAI/bge-reranker-base"))
    yield service
    await service.close()


class TestFastEmbedRerankerService:
    def test_load_model_uses_configured_cache_dir_without_cpu_memory_arena(
        self, reranker_service, monkeypatch
    ):
        cross_encoder_cls = MagicMock()
        monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", cross_encoder_cls)
        monkeypatch.setattr(reranker_service, "_model", None)

        reranker_service._load_model()

        cross_encoder_cls.assert_called_once_with(
            model_name=reranker_service.model_name,
            cache_dir=reranker_service.cache_dir,
            enable_cpu_mem_arena=False,
        )

    @pytest.mark.asyncio
    async def test_compute_score_sends_pairs_in_configured_batch_size(
        self, reranker_service, monkeypatch
    ):
        batch_sizes: list[int] = []

        class _Model:
            def rerank(self, query, documents, batch_size):
                batch_sizes.append(batch_size)
                return [0.0 for _ in documents]

        monkeypatch.setattr(reranker_service, "config", RerankerConfig(batch_size=3))
        monkeypatch.setattr(reranker_service, "_model", _Model())

        await reranker_service.compute_score([["query", f"doc-{i}"] for i in range(7)])

        assert batch_sizes == [3]

    @pytest.mark.asyncio
    async def test_compute_score_coerces_numpy_like_scores_to_float(
        self, reranker_service, monkeypatch
    ):
        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            MagicMock(__float__=lambda _self: 0.6),
            MagicMock(__float__=lambda _self: -0.1),
        ]
        monkeypatch.setattr(reranker_service, "_model", mock_model)

        scores = await reranker_service.compute_score(
            [
                ["query", "doc-a"],
                ["query", "doc-b"],
            ]
        )

        assert scores == [0.6, -0.1]

    @pytest.mark.asyncio
    async def test_compute_score_loads_and_scores_off_the_event_loop_thread(
        self, reranker_service, monkeypatch
    ):
        """首次重排时的模型加载与打分都在推理线程中执行。"""
        threads: list[int] = []

        def rerank(query, documents, batch_size):
            threads.append(threading.get_ident())
            return [0.0 for _ in documents]

        model = MagicMock()
        model.rerank.side_effect = rerank

        def load_model() -> None:
            threads.append(threading.get_ident())
            reranker_service._model = model

        monkeypatch.setattr(reranker_service, "_model", None)
        monkeypatch.setattr(reranker_service, "_load_model", load_model)

        assert await reranker_service.compute_score([["query", "doc"]]) == [0.0]

        assert len(threads) == 2
        assert threading.get_ident() not in threads
