"""
Rerank 服务基础模块

包含 Rerank 服务抽象接口和单例模式基类。
"""

import logging
import threading
from typing import List, Union, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRerankService(ABC):
    """
    Rerank 服务抽象接口
    
    定义所有 Rerank 服务必须实现的方法。
    """

    @abstractmethod
    def compute_score(
        self,
        pairs: List[List[str]],
        batch_size: int = 256,
        max_length: int = 512
    ) -> List[float]:
        """
        计算文本对的相似度分数
        
        Args:
            pairs: 文本对列表 [[query, passage], ...]
            batch_size: 批处理大小
            max_length: 最大长度
            
        Returns:
            分数列表
        """
        pass

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return True

    def warmup(self) -> None:
        """预热模型（默认无操作，子类可覆盖）"""
        pass

    def unload(self) -> None:
        """卸载模型以释放内存"""
        pass


class SingletonModelService(BaseRerankService):
    """
    单例模型服务基类
    
    封装了通用的单例模式、线程安全和延迟加载逻辑。
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """单例模式，确保全局只有一个实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: "RerankerConfig"
    ):
        """
        初始化服务配置
        
        Args:
            config: Reranker 配置对象
        """
        if self._initialized:
            return

        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.cache_dir = config.cache_dir
        self.use_fp16 = config.use_fp16
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

    def warmup(self) -> None:
        """
        预热模型 — 主动触发延迟加载

        在后台线程中调用此方法可避免首次请求时的加载延迟。
        线程安全：内部使用 _lazy_load_lock 保护。
        """
        if self._model is None:
            logger.info(f"{self.__class__.__name__} 开始预热模型: {self.model_name}")
            _ = self.model  # 触发 property 中的延迟加载
            logger.info(f"{self.__class__.__name__} 模型预热完成")

    def unload(self) -> None:
        """卸载模型以释放内存"""
        with self._lazy_load_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info(f"{self.__class__.__name__} 模型已卸载")
