"""
HiveMemory LLM 基础模块

包含 LLM 服务抽象接口和单例模式基类。
"""

import logging
import threading
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """
    LLM 服务抽象接口

    定义所有 LLM 服务必须实现的方法。
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成聊天补全

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            **kwargs: 其他额外参数

        Returns:
            str: LLM 响应内容
        """
        pass

    def complete_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs
    ) -> Optional[str]:
        """
        带重试机制的补全（默认实现，子类可覆盖）

        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            **kwargs: 其他参数

        Returns:
            Optional[str]: LLM 响应内容，失败时返回 None
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.complete(messages, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.warning(f"LLM 调用失败，重试耗尽: {e}")
        return None


class SingletonLLMService(BaseLLMService):
    """
    单例 LLM 服务基类

    封装了通用的单例模式、线程安全和配置管理。
    注意: LLM 服务是无状态的，不需要像 Embedding 那样的延迟加载模型。
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
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        初始化 LLM 服务配置

        Args:
            model: 模型名称 (如 "gpt-4o", "deepseek/deepseek-chat")
            api_key: API Key
            api_base: API Base URL
            temperature: 默认温度参数
            max_tokens: 默认最大 token 数
        """
        if self._initialized:
            return

        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._initialized = True

        logger.info(
            f"{self.__class__.__name__} 配置: "
            f"model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    def is_loaded(self) -> bool:
        """检查服务是否已初始化"""
        return self._initialized

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
