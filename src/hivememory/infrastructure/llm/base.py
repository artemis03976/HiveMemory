"""
HiveMemory LLM 基础模块

LLM 服务的中立接口与请求参数、调用结果、流式分块模型；本模块不依赖任何
提供商 SDK。

请求参数逐次携带模型与采样设置，实现不持有默认模型、凭证或采样参数；
使用方的固定配置在各自一侧转换为请求参数。提供商失败由实现在 SDK 边界
转换为 ``core.errors`` 的中立错误：上下文超窗为
``ModelContextWindowExceededError``，其他已知失败为 ``ModelProviderError``，
原始异常保留在原因链；程序错误与取消原样传播。服务接口不重试；需要对提供商失败
重试的使用方经 ``complete_with_provider_retry`` 调用。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Coroutine
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError

logger = logging.getLogger(__name__)

# 带重试的完整调用对提供商的最多请求次数（含首次）。
_MAX_ATTEMPTS = 2


class LLMRequestParams(BaseModel):
    """一次 LLM 调用的模型参数；为 ``None`` 或为空的可选项不发给提供商。

    ``api_key`` / ``api_base`` 为 ``None`` 是合法状态，由提供商 SDK 从环境变量
    读取凭证；``api_key`` 不出现在 repr 中，日志与异常格式化参数时不泄露密钥。
    ``json_output`` 要求提供商以 JSON 对象输出；提供商明确不支持该参数时，实现
    退回普通补全，由调用方按文本解析。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1, description="提供商可执行的模型标识符")
    api_key: str | None = Field(default=None, repr=False, description="本次调用的 API 密钥")
    api_base: str | None = Field(default=None, description="本次调用的 API 地址")
    temperature: float | None = Field(default=None, description="采样温度")
    top_p: float | None = Field(default=None, description="Top-p 采样率")
    max_tokens: int | None = Field(default=None, description="最大生成 token 数")
    stop: tuple[str, ...] = Field(default=(), description="停止序列")
    json_output: bool = Field(default=False, description="是否要求 JSON 对象输出")


class LLMUsage(BaseModel):
    """提供商报告的 token 用量。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_tokens: int = Field(description="本次调用消耗的总 token 数")
    prompt_tokens: int = Field(default=0, description="输入的 token 数；提供商没有报告时为 0")
    cached_prompt_tokens: int = Field(
        default=0,
        description="输入中命中提供商上下文缓存、按缓存价计费的 token 数；提供商没有报告时为 0",
    )


class LLMCompletion(BaseModel):
    """一次完整调用的结果；提供商没有返回文本时实现显式失败，不构造本结果。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(description="首个候选的文本内容")
    finish_reason: str | None = Field(
        default=None, description="提供商报告的结束原因；未报告时为 None，由调用方解释"
    )
    usage: LLMUsage | None = Field(default=None, description="提供商报告的用量，可缺省")


class LLMStreamChunk(BaseModel):
    """流式调用的一项。

    中间项只携带非空的文本增量；流正常耗尽时最后一项不带文本，携带提供商
    报告的结束原因（提供商未报告时为 ``None``）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: str = Field(default="", description="文本增量")
    finish_reason: str | None = Field(default=None, description="结束原因，只出现在最后一项")


class BaseLLMService(ABC):
    """
    LLM 服务抽象接口

    调用方只面对唯一入口 :meth:`complete`，以 ``stream`` 参数选择完整调用或
    流式调用；实现只提供两个受保护的钩子，不感知调用方是谁。
    """

    @overload
    def complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: Literal[False] = False,
    ) -> Coroutine[Any, Any, LLMCompletion]: ...

    @overload
    def complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: Literal[True],
    ) -> AsyncGenerator[LLMStreamChunk, None]: ...

    def complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: bool = False,
    ) -> Coroutine[Any, Any, LLMCompletion] | AsyncGenerator[LLMStreamChunk, None]:
        """``stream=False``（默认）返回可 await 的完整结果，``stream=True`` 返回流。

        流首次拉取时才发出请求；流耗尽、被调用方关闭或在拉取时被取消，都会
        关闭底层响应，关闭失败只记录日志，不替换正在传播的异常或取消。
        """
        if stream:
            return self._stream(messages, params)
        return self._complete(messages, params)

    @abstractmethod
    async def _complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> LLMCompletion:
        """执行一次完整调用并返回首个候选的文本。"""

    @abstractmethod
    def _stream(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """返回流式调用的异步生成器，语义见 :meth:`complete`。"""


async def complete_with_provider_retry(
    llm_service: BaseLLMService, messages: list[dict[str, str]], params: LLMRequestParams
) -> LLMCompletion:
    """完整调用；只对提供商失败重试一次，最后一次的失败原样抛出。

    服务接口本身不重试，是否重试由使用方决定。超窗时同一请求必然再次失败，不重试；
    程序错误与取消不重试、原样传播。
    """
    for attempt in range(1, _MAX_ATTEMPTS):
        try:
            return await llm_service.complete(messages, params)
        except ModelContextWindowExceededError:
            raise
        except ModelProviderError as error:
            logger.warning("LLM 调用失败 (尝试 %s/%s): %s", attempt, _MAX_ATTEMPTS, error)
    return await llm_service.complete(messages, params)


__all__ = [
    "BaseLLMService",
    "LLMCompletion",
    "LLMRequestParams",
    "LLMStreamChunk",
    "LLMUsage",
    "complete_with_provider_retry",
]
