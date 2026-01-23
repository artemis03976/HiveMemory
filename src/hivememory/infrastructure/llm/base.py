"""
HiveMemory LLM 基础模块

包含 LLM 服务抽象接口和单例模式基类。
"""

import logging
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from hivememory.patchouli.config import LLMConfig

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

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        生成聊天补全（支持 Function Calling）

        用于需要返回完整响应对象的场景（如 Function Calling）。

        Args:
            messages: 消息列表
            tools: Function Calling 工具定义列表
            tool_choice: 工具选择策略
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数

        Returns:
            Any: 完整的 LLM 响应对象（包含 tool_calls 等信息）

        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support complete_with_tools")

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
        config: "LLMConfig",
    ):
        """
        初始化 LLM 服务配置

        Args:
            config: LLM 配置对象
        """
        if self._initialized:
            return

        self.config = config
        self.model = config.model
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        self._initialized = True

        logger.info(
            f"{self.__class__.__name__} 配置: "
            f"model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}"
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
