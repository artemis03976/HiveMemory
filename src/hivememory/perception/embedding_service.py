"""
HiveMemory 本地 Embedding 服务

使用 sentence-transformers 库加载轻量级模型，
用于感知层的语义吸附与漂移检测。

参考: PROJECT.md 8.3 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
import threading
from typing import List, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """
    本地 Embedding 服务

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
        texts: str | List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[float] | List[List[float]]:
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
        text2: str | List[str]
    ) -> float | List[float]:
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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> LocalEmbeddingService:
    """
    获取全局 Embedding 服务实例（单例）

    Args:
        model_name: 模型名称
        device: 运行设备
        cache_dir: 模型缓存目录

    Returns:
        LocalEmbeddingService: 全局单例实例

    Examples:
        >>> from hivememory.perception.embedding_service import get_embedding_service
        >>> service = get_embedding_service()
        >>> vector = service.encode("hello")
    """
    # 使用 hashable 的参数组合来支持缓存
    cache_key = (model_name, device, cache_dir)
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
        >>> from hivememory.perception.embedding_service import create_embedding_service
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


__all__ = [
    "LocalEmbeddingService",
    "get_embedding_service",
    "create_embedding_service",
]
