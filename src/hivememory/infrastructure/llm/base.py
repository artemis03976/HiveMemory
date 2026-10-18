"""
HiveMemory LLM 基础模块

包含 LLM 服务抽象接口和绑定配置的服务基类。
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hivememory.config.shared import LLMConfig

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """
    LLM 服务抽象接口

    定义所有 LLM 服务必须实现的方法。
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
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
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
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

    async def acomplete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Any:
        """
        异步生成聊天补全（支持 Function Calling）

        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support acomplete_with_tools"
        )

    async def acomplete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """
        异步生成 JSON 内容。

        子类可使用 provider 的 JSON mode / structured output 能力实现；
        不支持时应降级为普通 completion 并返回文本内容。
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support acomplete_json")

    def complete_with_retry(
        self, messages: list[dict[str, str]], max_retries: int = 2, **kwargs
    ) -> str | None:
        """
        带重试机制的补全（默认实现，子类可覆盖）

        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            **kwargs: 其他参数

        Returns:
            Optional[str]: LLM 响应内容，失败时返回 None
        """
        for attempt in range(max_retries):
            try:
                return self.complete(messages, **kwargs)
            except Exception as e:
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.warning(f"LLM 调用失败，重试耗尽: {e}")
        return None


class ConfiguredLLMService(BaseLLMService):
    """
    绑定一份 LLM 配置的服务基类。

    每个实例只持有构造时的配置；不同组件（如 Gateway 与 Librarian）各自构造
    实例，彼此的模型、凭证与采样参数互不影响。服务本身无状态，不需要共享实例。
    """

    def __init__(
        self,
        config: "LLMConfig",
    ):
        """
        初始化 LLM 服务配置

        Args:
            config: LLM 配置对象
        """
        self.config = config
        self.model = config.model
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        logger.info(
            f"{self.__class__.__name__} 配置: "
            f"model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def get_config(self) -> dict[str, Any]:
        """获取当前配置"""
        return {
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
