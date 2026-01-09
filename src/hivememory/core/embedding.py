"""
HiveMemory 本地 Embedding 服务

使用 sentence-transformers 库加载轻量级模型，
用于感知层的语义吸附、漂移检测以及存储层的向量生成。

参考: PROJECT.md 8.3 节

作者: HiveMemory Team
版本: 1.2.0
"""

import logging
import threading
from typing import List, Optional, Tuple, Union, Dict, Any
from functools import lru_cache
from abc import ABC, abstractmethod

from hivememory.core.config import get_config

logger = logging.getLogger(__name__)


# ========== Embedding Service 抽象接口 ==========

class BaseEmbeddingService(ABC):
    """
    Embedding 服务抽象接口

    定义所有 embedding 服务必须实现的方法。
    """

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化
            show_progress: 是否显示进度条

        Returns:
            向量或向量列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return True

    def unload(self) -> None:
        """卸载模型以释放内存"""
        pass


# ========== SentenceTransformers Embedding 服务 ==========


class LocalEmbeddingService(BaseEmbeddingService):
    """
    本地 Embedding 服务 (SentenceTransformers)

    使用 sentence-transformers 库加载轻量级模型。
    采用单例模式，支持延迟初始化。

    默认模型：all-MiniLM-L6-v2
    - 大小：约 80MB
    - 向量维度：384
    - 推理速度：CPU 上毫秒级

    Examples:
        >>> service = LocalEmbeddingService()
        >>> vector = service.encode("你好世界")
        >>> print(len(vector))  # 384
        >>>
        >>> # 计算相似度
        >>> similarity = service.compute_similarity("你好", "您好")
        >>> print(similarity)  # 0.8...
    """

    _instance: Optional["LocalEmbeddingService"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ) -> "LocalEmbeddingService":
        """单例模式，确保全局只有一个 Embedding 服务实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        初始化本地 Embedding 服务

        Args:
            model_name: 模型名称（HuggingFace 模型 ID）
            device: 运行设备 ("cpu" 或 "cuda"）
            cache_dir: 模型缓存目录（可选）
        """
        if self._initialized:
            return

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._model_lock = threading.Lock()
        self._initialized = True

        logger.info(
            f"LocalEmbeddingService 配置: "
            f"model={model_name}, device={device}"
        )

    @property
    def model(self):
        """延迟加载模型，首次使用时才加载"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self) -> None:
        """加载 sentence-transformers 模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请运行: pip install sentence-transformers"
            )

        logger.info(f"正在加载 Embedding 模型: {self.model_name}")
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成，向量维度: {dim}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化（用于余弦相似度）
            show_progress: 是否显示进度条

        Returns:
            向量或向量列表

        Examples:
            >>> service = LocalEmbeddingService()
            >>> vec = service.encode("你好世界")
            >>> print(len(vec))  # 384
            >>>
            >>> vecs = service.encode(["你好", "世界"])
            >>> print(len(vecs))  # 2
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress
            )
        except Exception as e:
            logger.warning(f"Encoding 失败，使用零向量: {e}")
            dim = self.model.get_sentence_embedding_dimension()
            embeddings = __import__("numpy").zeros((len(texts), dim))

        if single_input:
            return embeddings[0].tolist()
        return embeddings.tolist()

    def compute_similarity(
        self,
        text1: str,
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """
        计算余弦相似度

        Args:
            text1: 查询文本
            text2: 目标文本或文本列表

        Returns:
            相似度 (0-1) 或相似度列表

        Examples:
            >>> service = LocalEmbeddingService()
            >>> similarity = service.compute_similarity("你好", "您好")
            >>> print(similarity)  # 0.8...
            >>>
            >>> similarities = service.compute_similarity("你好", ["您好", "再见"])
            >>> print(similarities)  # [0.8..., 0.2...]
        """
        vec1 = self.encode(text1, normalize=True)

        single_target = isinstance(text2, str)
        if single_target:
            text2 = [text2]

        vecs2 = self.encode(text2, normalize=True)

        import numpy as np

        vec1_array = __import__("numpy").array(vec1)
        vecs2_array = __import__("numpy").array(vecs2)

        similarities = np.dot(vecs2_array, vec1_array)

        if single_target:
            return float(similarities)
        return similarities.tolist()

    def get_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        if self._model is None:
            # 预设默认维度
            if "all-MiniLM-L6-v2" in self.model_name:
                return 384
            elif "bge-small" in self.model_name:
                return 384
            elif "bge-base" in self.model_name:
                return 768
            else:
                return 384
        return self.model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 模型是否已加载
        """
        return self._model is not None

    def unload(self) -> None:
        """
        卸载模型以释放内存

        注意：卸载后下次使用时会自动重新加载
        """
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info("Embedding 模型已卸载")


@lru_cache(maxsize=1)
def get_embedding_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> LocalEmbeddingService:
    """
    获取全局 Embedding 服务实例（单例）

    如果未指定参数，则使用 hivememory.core.config 中的全局配置。

    Args:
        model_name: 模型名称 (可选)
        device: 运行设备 (可选)
        cache_dir: 模型缓存目录 (可选)

    Returns:
        LocalEmbeddingService: 全局单例实例

    Examples:
        >>> from hivememory.core.embedding import get_embedding_service
        >>> service = get_embedding_service()
        >>> vector = service.encode("hello")
    """
    if model_name is None:
        config = get_config().embedding
        model_name = config.model_name
        # 仅在参数未提供时使用配置中的默认值
        if device is None:
            device = config.device
        if cache_dir is None:
            cache_dir = config.cache_dir
    
    # 确保有默认值
    if device is None:
        device = "cpu"

    # 使用 hashable 的参数组合来支持缓存
    return LocalEmbeddingService(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )


def create_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> LocalEmbeddingService:
    """
    创建新的 Embedding 服务实例（不使用单例）

    用于需要多个不同模型实例的场景。

    Args:
        model_name: 模型名称
        device: 运行设备
        cache_dir: 模型缓存目录

    Returns:
        LocalEmbeddingService: 新的服务实例

    Examples:
        >>> from hivememory.core.embedding import create_embedding_service
        >>> service = create_embedding_service(model_name="bge-small-en-v1.5")
    """
    # 重置单例标记以创建新实例
    instance = object.__new__(LocalEmbeddingService)
    instance._initialized = False
    instance.__init__(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )
    return instance


# ========== BGE-M3 Hybrid Embedding 服务 ==========


class BGEM3EmbeddingService(BaseEmbeddingService):
    """
    BGE-M3 混合 Embedding 服务

    BGE-M3 是一个多功能模型，支持:
    - Dense Vector: 语义向量 (维度 1024)
    - Sparse Vector: 稀疏向量 (词汇权重，类似 BM25)
    - Multi-Vector: ColBERT 风格向量 (暂不使用)

    用于混合检索场景，同时生成稠密和稀疏向量。

    模型: BAAI/bge-m3
    - Dense 维度: 1024
    - Sparse: token_id -> weight 的字典

    Examples:
        >>> service = BGEM3EmbeddingService()
        >>> result = service.encode("Python 日期处理代码")
        >>> print(result["dense"][:5])  # Dense vector
        >>> print(list(result["sparse"].keys())[:5])  # Sparse tokens
    """

    _instance: Optional["BGEM3EmbeddingService"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ) -> "BGEM3EmbeddingService":
        """单例模式，确保全局只有一个 BGE-M3 服务实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        初始化 BGE-M3 Embedding 服务

        Args:
            model_name: 模型名称 (默认 BAAI/bge-m3)
            device: 运行设备 ("cpu" 或 "cuda"）
            cache_dir: 模型缓存目录（可选）
        """
        if self._initialized:
            return

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._model_lock = threading.Lock()
        self._initialized = True

        logger.info(
            f"BGEM3EmbeddingService 配置: "
            f"model={model_name}, device={device}"
        )

    @property
    def model(self):
        """延迟加载模型，首次使用时才加载"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self) -> None:
        """加载 BGE-M3 模型"""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding 未安装。请运行: pip install FlagEmbedding"
            )

        logger.info(f"正在加载 BGE-M3 模型: {self.model_name}")
        try:
            self._model = BGEM3FlagModel(
                model_name_or_path=self.model_name,
                device=self.device,
                use_fp16=self.device != "cpu",
                show_progress_bar=False,
            )
            logger.info("BGE-M3 模型加载完成")
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise

    def encode(
        self,
        dense_texts: Union[str, List[str], None] = None,
        sparse_texts: Union[str, List[str], None] = None,
    ) -> Union[List[float], Dict[int, float], Dict[str, Any], List[Dict[str, Any]]]:
        """
        编码文本为稠密向量、稀疏向量或混合向量

        根据传入的参数返回对应的结果：
        - 仅传入 dense_texts: 返回稠密向量
        - 仅传入 sparse_texts: 返回稀疏向量
        - 两者都传入: 返回 {"dense": ..., "sparse": ...}

        Args:
            dense_texts: 用于稠密向量的文本（单个或列表）
            sparse_texts: 用于稀疏向量的文本（单个或列表，通常包含更多关键词）

        Returns:
            仅 dense: List[float] 或 List[List[float]]
            仅 sparse: Dict[int, float] 或 List[Dict[int, float]]
            两者都有: {"dense": List[float], "sparse": Dict[int, float]]}

        Examples:
            >>> service = BGEM3EmbeddingService()
            >>> # 仅稠密
            >>> dense = service.encode(dense_texts="Python 日期处理")
            >>> # 仅稀疏
            >>> sparse = service.encode(sparse_texts="Python datetime date now")
            >>> # 混合（使用不同输入）
            >>> result = service.encode(dense_texts="简短描述", sparse_texts="完整上下文关键词")
            >>> dense = result["dense"]
            >>> sparse = result["sparse"]
        """
        import numpy as np

        # 至少需要一种输入
        if dense_texts is None and sparse_texts is None:
            raise ValueError("至少需要提供 dense_texts 或 sparse_texts 参数")

        # 处理稠密向量
        dense_result = None
        if dense_texts is not None:
            single_dense_input = isinstance(dense_texts, str)
            dense_input_list = [dense_texts] if single_dense_input else dense_texts

            try:
                output = self.model.encode(
                    dense_input_list,
                    batch_size=1,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense_vecs = output['dense_vecs']
                if single_dense_input:
                    dense_result = dense_vecs[0].tolist()
                else:
                    dense_result = dense_vecs.tolist()
            except Exception as e:
                logger.warning(f"稠密向量编码失败: {e}")
                size = 1 if single_dense_input else len(dense_input_list)
                dense_result = np.zeros((size, 1024)).tolist() if not single_dense_input else []

        # 处理稀疏向量
        sparse_result = None
        if sparse_texts is not None:
            single_sparse_input = isinstance(sparse_texts, str)
            sparse_input_list = [sparse_texts] if single_sparse_input else sparse_texts

            try:
                output = self.model.encode(
                    sparse_input_list,
                    batch_size=1,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    verbose=False  # Disable progress bar
                )
                lexical_weights = output['lexical_weights']

                if single_sparse_input:
                    sparse_weights = lexical_weights[0]
                    sparse_result = {
                        int(k): float(v) for k, v in sparse_weights.items()
                    }
                else:
                    sparse_result = []
                    for weights in lexical_weights:
                        sparse_result.append({
                            int(k): float(v) for k, v in weights.items()
                        })
            except Exception as e:
                logger.warning(f"稀疏向量编码失败: {e}")
                sparse_result = {} if single_sparse_input else []

        # 返回结果
        if dense_result is not None and sparse_result is not None:
            return {"dense": dense_result, "sparse": sparse_result}
        elif dense_result is not None:
            return dense_result
        else:
            return sparse_result

    def get_dimension(self) -> int:
        """
        获取稠密向量维度

        Returns:
            int: BGE-M3 稠密向量维度 (1024)
        """
        return 1024

    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 模型是否已加载
        """
        return self._model is not None

    def unload(self) -> None:
        """
        卸载模型以释放内存

        注意：卸载后下次使用时会自动重新加载
        """
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info("BGE-M3 模型已卸载")


@lru_cache(maxsize=1)
def get_bge_m3_service(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> BGEM3EmbeddingService:
    """
    获取全局 BGE-M3 服务实例（单例）

    Args:
        model_name: 模型名称 (默认 BAAI/bge-m3)
        device: 运行设备 (可选)
        cache_dir: 模型缓存目录 (可选)

    Returns:
        BGEM3EmbeddingService: 全局单例实例

    Examples:
        >>> from hivememory.core.embedding import get_bge_m3_service
        >>> service = get_bge_m3_service()
        >>> result = service.encode("hello world")
    """
    return BGEM3EmbeddingService(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )


__all__ = [
    "BaseEmbeddingService",
    "LocalEmbeddingService",
    "BGEM3EmbeddingService",
    "get_embedding_service",
    "get_bge_m3_service",
    "create_embedding_service",
]
