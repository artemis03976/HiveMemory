"""
HiveMemory Embedding 基础模块

包含 Embedding 服务抽象接口和单例模式基类。
"""

import logging
import threading
from typing import List, Union, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from hivememory.patchouli.config import EmbeddingConfig

logger = logging.getLogger(__name__)


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

    def compute_cosine_similarity(
        self,
        vector_a: List[float],
        vector_b: List[float]
    ) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector_a: 向量 A
            vector_b: 向量 B
            
        Returns:
            float: 余弦相似度 (-1.0 ~ 1.0)
        """
        import numpy as np
        
        if not vector_a or not vector_b:
            return 0.0
            
        try:
            array_a = np.array(vector_a)
            array_b = np.array(vector_b)
            
            norm_a = np.linalg.norm(array_a)
            norm_b = np.linalg.norm(array_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return float(np.dot(array_a, array_b) / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return 0.0

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return True

    def unload(self) -> None:
        """卸载模型以释放内存"""
        pass


class SingletonModelService(BaseEmbeddingService):
    """
    单例模型服务基类 (支持多例模式 Multiton)
    
    封装了通用的单例/多例模式、线程安全和延迟加载逻辑。
    基于配置的 model_name 和 device 区分实例。
    """
    _instances = {}
    _lock = threading.Lock()
    _model_lock = threading.Lock()

    def __new__(cls, config: "EmbeddingConfig", *args, **kwargs):
        """
        Multiton 模式: 根据配置的 model_name 和 device 获取或创建实例
        """
        # 生成唯一键 (使用 model_name 和 device 作为键)
        key = f"{config.model_name}:{config.device}"

        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        config: "EmbeddingConfig",
    ):
        """
        初始化服务配置
        
        Args:
            config: Embedding 配置对象
        """
        if getattr(self, "_initialized", False):
            return

        self.config = config
        self.model_name = self.config.model_name
        self.device = self.config.device
        self.cache_dir = self.config.cache_dir
        self._model = None

        self._lazy_load_lock = threading.Lock()
        
        self._initialized = True
        
        logger.info(
            f"{self.__class__.__name__} 配置: "
            f"model={self.model_name}, device={self.device}"
        )

    @property
    def model(self):
        """延迟加载模型，首次使用时才加载"""
        if self._model is None:
            with self._lazy_load_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> None:
        """加载具体模型的逻辑，由子类实现"""
        pass

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

    def unload(self) -> None:
        """卸载模型以释放内存"""
        with self._lazy_load_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info(f"{self.__class__.__name__} 模型已卸载")
