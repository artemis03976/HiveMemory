"""
HiveMemory LiteLLM 服务实现

封装 litellm.completion() 调用，支持重试机制。
"""

import logging
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

import litellm

from hivememory.core.llm.base import SingletonLLMService

logger = logging.getLogger(__name__)


class LiteLLMService(SingletonLLMService):
    """
    LiteLLM 服务实现

    使用 litellm 库统一调用各种 LLM 提供商。

    特性:
        - 统一的 API 接口
        - 自动重试机制
        - 可选的流式输出支持（未来扩展）
        - 详细的日志记录
    """

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
            messages: 消息列表
            temperature: 温度参数（None 则使用实例默认值）
            max_tokens: 最大 token 数（None 则使用实例默认值）
            **kwargs: 传递给 litellm.completion 的额外参数

        Returns:
            str: LLM 响应内容

        Raises:
            Exception: LLM 调用失败时抛出
        """
        logger.debug(f"调用 LLM: {self.model}, 消息数: {len(messages)}")

        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )

        content = response.choices[0].message.content

        # 记录 token 使用情况
        if hasattr(response, 'usage') and response.usage:
            logger.info(
                f"LLM 调用成功 (model={self.model}, "
                f"tokens={response.usage.total_tokens})"
            )
        else:
            logger.info(f"LLM 调用成功 (model={self.model})")

        logger.debug(f"LLM 响应长度: {len(content)} 字符")

        return content

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

        用于 Global Gateway 等需要使用 Function Calling 的场景。
        返回完整的响应对象，包含 tool_calls 等信息。

        Args:
            messages: 消息列表
            tools: Function Calling 工具定义列表
            tool_choice: 工具选择策略 (如 {"type": "function", "function": {"name": "..."}})
            temperature: 温度参数（None 则使用实例默认值）
            max_tokens: 最大 token 数（None 则使用实例默认值）
            **kwargs: 其他传递给 litellm.completion 的参数

        Returns:
            Any: 完整的 litellm 响应对象

        Raises:
            Exception: LLM 调用失败时抛出

        Examples:
            >>> response = service.complete_with_tools(
            ...     messages=[{"role": "user", "content": "What's the weather?"}],
            ...     tools=[{"type": "function", "function": {...}}],
            ...     tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ... )
            >>> if response.choices[0].message.tool_calls:
            ...     tool_call = response.choices[0].message.tool_calls[0]
            ...     args = json.loads(tool_call.function.arguments)
        """
        logger.debug(
            f"调用 LLM (with tools): {self.model}, "
            f"消息数: {len(messages)}, 工具数: {len(tools) if tools else 0}"
        )

        # 构建 litellm 参数
        llm_params = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        # 添加 Function Calling 参数
        if tools:
            llm_params["tools"] = tools
        if tool_choice:
            llm_params["tool_choice"] = tool_choice

        # 添加额外的 kwargs
        llm_params.update(kwargs)

        response = litellm.completion(**llm_params)

        # 记录 token 使用情况
        if hasattr(response, 'usage') and response.usage:
            logger.info(
                f"LLM 调用成功 (model={self.model}, "
                f"tokens={response.usage.total_tokens}, "
                f"tool_calls={bool(response.choices[0].message.tool_calls) if response.choices else False})"
            )
        else:
            logger.info(f"LLM 调用成功 (model={self.model})")

        return response

    def complete_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs
    ) -> Optional[str]:
        """
        带重试机制的补全（覆盖基类实现，添加更详细的日志）

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
                logger.debug(f"调用 LLM (尝试 {attempt + 1}/{max_retries})...")
                result = self.complete(messages, **kwargs)
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                # 最后一次尝试失败时才记录警告
                if attempt == max_retries - 1:
                    logger.warning(f"LLM 调用失败，重试耗尽: {e}")

        return None


# ========== 工厂函数 ==========

@lru_cache(maxsize=1)
def get_worker_llm_service(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LiteLLMService:
    """
    获取 Worker LLM 服务实例（单例）

    用于 ChatBot 等对话场景。

    Args:
        model: 模型名称（None 则使用配置文件默认值）
        api_key: API Key（None 则使用配置文件默认值）
        api_base: API Base URL（None 则使用配置文件默认值）
        temperature: 温度参数（None 则使用配置文件默认值）
        max_tokens: 最大 token 数（None 则使用配置文件默认值）

    Returns:
        LiteLLMService: Worker LLM 服务实例

    Examples:
        >>> service = get_worker_llm_service()
        >>> response = service.complete([{"role": "user", "content": "Hello"}])
    """
    from hivememory.core.config import load_app_config

    # 从配置获取默认值
    config = load_app_config().get_worker_llm_config()

    if model is None:
        model = config.model
    if api_key is None:
        api_key = config.api_key
    if api_base is None:
        api_base = config.api_base
    if temperature is None:
        temperature = config.temperature
    if max_tokens is None:
        max_tokens = config.max_tokens

    return LiteLLMService(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_librarian_llm_service(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LiteLLMService:
    """
    获取 Librarian LLM 服务实例（单例）

    用于记忆提取等任务。

    Args:
        model: 模型名称（None 则使用配置文件默认值）
        api_key: API Key（None 则使用配置文件默认值）
        api_base: API Base URL（None 则使用配置文件默认值）
        temperature: 温度参数（None 则使用配置文件默认值）
        max_tokens: 最大 token 数（None 则使用配置文件默认值）

    Returns:
        LiteLLMService: Librarian LLM 服务实例

    Examples:
        >>> service = get_librarian_llm_service()
        >>> response = service.complete([{"role": "user", "content": "..."}])
    """
    from hivememory.core.config import load_app_config

    # 从配置获取默认值
    config = load_app_config().get_librarian_llm_config()

    if model is None:
        model = config.model
    if api_key is None:
        api_key = config.api_key
    if api_base is None:
        api_base = config.api_base
    if temperature is None:
        temperature = config.temperature
    if max_tokens is None:
        max_tokens = config.max_tokens

    return LiteLLMService(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = [
    "LiteLLMService",
    "get_worker_llm_service",
    "get_librarian_llm_service",
]
