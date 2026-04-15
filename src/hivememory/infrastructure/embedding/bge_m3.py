"""
HiveMemory BGE-M3 Dense Embedding 服务 (FastEmbed ONNX 实现)
"""

import logging
import threading
from typing import List, Union, Dict, Any, Optional, TYPE_CHECKING

from hivememory.patchouli.config import load_app_config
if TYPE_CHECKING:
    from hivememory.patchouli.config import EmbeddingConfig
from hivememory.infrastructure.embedding.base import SingletonModelService

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
            from fastembed.common.model_description import PoolingType, ModelSource
        except ImportError:
            raise ImportError(
                "fastembed 未安装。请运行: pip install fastembed"
            )

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

    def encode(
        self,
        dense_texts: Union[str, List[str], None] = None,
        sparse_texts: Union[str, List[str], None] = None,
        **kwargs
    ) -> Union[List[float], str, Dict[str, Any]]:
        """
        编码文本为稠密向量，或返回 sparse 原始文本供 Qdrant BM25 使用。

        Returns:
            - dense only: List[float]
            - sparse only: str (原始文本)
            - both: {"dense": List[float], "sparse_text": str}
        """
        if dense_texts is None and sparse_texts is None:
            if "texts" in kwargs:
                dense_texts = kwargs["texts"]
            else:
                raise ValueError("至少需要提供 dense_texts 或 sparse_texts 参数")

        dense_result = None
        if dense_texts is not None:
            single = isinstance(dense_texts, str)
            input_list = [dense_texts] if single else dense_texts
            try:
                embeddings = list(self.model.embed(input_list))
                dense_result = embeddings[0].tolist() if single else [e.tolist() for e in embeddings]
            except Exception as e:
                logger.warning(f"稠密向量编码失败: {e}")
                dense_result = [] if single else [[] for _ in input_list]

        sparse_result = None
        if sparse_texts is not None:
            sparse_result = sparse_texts

        if dense_result is not None and sparse_result is not None:
            return {"dense": dense_result, "sparse_text": sparse_result}
        if dense_result is not None:
            return dense_result
        return sparse_result

    def get_dimension(self) -> int:
        """获取稠密向量维度"""
        return _CUSTOM_BGE_M3_DIM


_bge_m3_instance = None
_bge_m3_lock = threading.Lock()


def get_bge_m3_service(
    config: Optional["EmbeddingConfig"] = None
) -> BGEM3EmbeddingService:
    """获取全局 BGE-M3 服务实例（单例）"""
    global _bge_m3_instance

    if _bge_m3_instance is None:
        with _bge_m3_lock:
            if _bge_m3_instance is None:
                if config is None:
                    global_embedding_config = load_app_config().embedding
                    base_config = global_embedding_config.default
                    config = base_config.model_copy(update={"model_name": _CUSTOM_BGE_M3_MODEL})

                _bge_m3_instance = BGEM3EmbeddingService(config=config)

    return _bge_m3_instance
