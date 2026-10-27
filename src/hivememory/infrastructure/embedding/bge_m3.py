"""
HiveMemory BGE-M3 Dense Embedding 服务 (FastEmbed ONNX 实现)
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional, overload

if TYPE_CHECKING:
    from hivememory.config.shared import EmbeddingConfig
from hivememory.infrastructure.embedding.base import (
    DenseSparseEncoding,
    EmbeddingOutput,
    SingletonModelService,
)

logger = logging.getLogger(__name__)

_CUSTOM_BGE_M3_MODEL = "Xenova/bge-m3"
_CUSTOM_BGE_M3_FILE = "onnx/model_int8.onnx"
_CUSTOM_BGE_M3_DIM = 1024


class BGEM3EmbeddingService(SingletonModelService):
    """
    BGE-M3 Dense Embedding 服务 (FastEmbed ONNX)

    支持:
    - Dense Vector: 语义向量 (维度 1024)
    - Sparse: 返回原始文本供 Qdrant BM25 使用
    """

    def _load_model(self) -> None:
        """加载 FastEmbed BGE-M3 ONNX 模型"""
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError:
            raise ImportError("fastembed 未安装。请运行: pip install fastembed")

        logger.info(f"正在加载 FastEmbed BGE-M3 模型: {self.model_name}")
        try:
            TextEmbedding.add_custom_model(
                model=_CUSTOM_BGE_M3_MODEL,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf=_CUSTOM_BGE_M3_MODEL),
                dim=_CUSTOM_BGE_M3_DIM,
                model_file=_CUSTOM_BGE_M3_FILE,
            )
        except ValueError:
            pass

        try:
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
            logger.info("FastEmbed BGE-M3 模型加载完成")
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise

    # 存储层的两种调用（单条 dense、单条 dense+sparse）有精确返回类型；
    # 其余组合按通用签名返回联合。
    @overload
    async def encode(self, dense_texts: str, sparse_texts: None = None) -> list[float]: ...

    @overload
    async def encode(self, dense_texts: str, sparse_texts: str) -> DenseSparseEncoding: ...

    @overload
    async def encode(
        self,
        dense_texts: str | list[str] | None = None,
        sparse_texts: str | list[str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingOutput: ...

    async def encode(
        self,
        dense_texts: str | list[str] | None = None,
        sparse_texts: str | list[str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """
        编码文本为稠密向量，或返回 sparse 原始文本供 Qdrant BM25 使用。

        稠密编码（含首次使用时的模型加载）在推理线程中执行。

        Returns:
            - dense only: List[float]
            - sparse only: str (原始文本)
            - both: {"dense": List[float], "sparse_text": str}
        """
        if dense_texts is None and sparse_texts is None:
            dense_texts = kwargs.get("texts")

        if dense_texts is None:
            if sparse_texts is None:
                raise ValueError("至少需要提供 dense_texts 或 sparse_texts 参数")
            # sparse only：返回原文，由向量库计算 BM25
            return sparse_texts

        dense_result = await self._inference.run(self._embed_dense, dense_texts)
        if sparse_texts is None:
            return dense_result
        return {"dense": dense_result, "sparse_text": sparse_texts}

    def _embed_dense(self, dense_texts: str | list[str]) -> list[float] | list[list[float]]:
        """在推理线程中计算稠密向量；模型失败时退化为空向量。"""
        single = isinstance(dense_texts, str)
        input_list = [dense_texts] if isinstance(dense_texts, str) else dense_texts
        try:
            embeddings = list(self.model.embed(input_list))
            return embeddings[0].tolist() if single else [e.tolist() for e in embeddings]
        except Exception as e:
            logger.warning(f"稠密向量编码失败: {e}")
            empty_batch: list[list[float]] = [[] for _ in input_list]
            return [] if single else empty_batch


_bge_m3_instance = None
_bge_m3_lock = threading.Lock()


def get_bge_m3_service(config: Optional["EmbeddingConfig"] = None) -> BGEM3EmbeddingService:
    """获取全局 BGE-M3 服务实例（单例）"""
    global _bge_m3_instance

    if _bge_m3_instance is None:
        with _bge_m3_lock:
            if _bge_m3_instance is None:
                # 配置由调用方（存储层装配）显式提供；基础设施层不读取全局配置。
                if config is None:
                    raise ValueError("config is required for get_bge_m3_service")

                _bge_m3_instance = BGEM3EmbeddingService(config=config)

    return _bge_m3_instance
