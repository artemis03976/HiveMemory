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

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return True

    def unload(self) -> None:
        """卸载模型以释放内存"""
        pass


class SingletonModelService(BaseEmbeddingService):
    """
    单例模型服务基类
    
    封装了通用的单例模式、线程安全和延迟加载逻辑。
    """
    _instance = None
    _lock = threading.Lock()
    _model_lock = threading.Lock()
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
        config: "EmbeddingConfig",
    ):
        """
        初始化服务配置
        
        Args:
            config: Embedding 配置对象
        """
        if self._initialized:
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
