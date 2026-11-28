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

请求可以携带工具，让模型以原生工具调用作答；完整调用的结果交回工具调用与提供商返回的
assistant 消息原样，调用方把这条消息追加进对话、为每个调用交回一条 ``tool`` 消息，再发
下一次请求。不支持工具调用的模型由提供商拒绝请求，以提供商失败结束。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Coroutine
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError

logger = logging.getLogger(__name__)

# 带重试的完整调用对提供商的最多请求次数（含首次）。
_MAX_ATTEMPTS = 2


class LLMTool(BaseModel):
    """请求携带的一个工具：模型按名称调用它，参数按 JSON Schema 给出。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="工具名，模型调用时原样交回")
    description: str = Field(description="一句话说明工具做什么")
    parameters: dict[str, Any] = Field(description="参数的 JSON Schema，顶层是 object")


class LLMToolCall(BaseModel):
    """模型返回的一个工具调用；参数保持原文，``parsed_arguments`` 按约定解析。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="调用 id，交回结果的 tool 消息以它对应这个调用")
    name: str = Field(description="被调用的工具名")
    arguments: str = Field(description="参数原文，按约定是 JSON 对象，但不保证合法")

    def parsed_arguments(self) -> dict[str, Any] | None:
        """参数原文解析成的 JSON 对象；不是合法的 JSON、或顶层不是对象时为 ``None``。"""
        try:
            arguments = json.loads(self.arguments)
        except json.JSONDecodeError:
            return None
        return arguments if isinstance(arguments, dict) else None


def tool_result_message(call: LLMToolCall, content: str) -> dict[str, Any]:
    """回应一个工具调用的 ``tool`` 消息；模型服务要求对话中每个调用 id 都有一条。"""
    return {"role": "tool", "tool_call_id": call.id, "content": content}


class LLMRequestParams(BaseModel):
    """一次 LLM 调用的模型参数；为 ``None`` 或为空的可选项不发给提供商。

    ``api_key`` / ``api_base`` 为 ``None`` 是合法状态，由提供商 SDK 从环境变量
    读取凭证；``api_key`` 不出现在 repr 中，日志与异常格式化参数时不泄露密钥。
    ``json_output`` 要求提供商以 JSON 对象输出；提供商明确不支持该参数时，实现
    退回普通补全，由调用方按文本解析。

    带 ``tools`` 时由模型决定调用哪些工具（``tool_choice`` 固定为 ``auto``，不开放配置：
    DeepSeek 在思考模式下不接受 ``required``，``auto`` 对各家都成立）。工具调用与 JSON 对象
    输出是两种互斥的作答方式，同时给出时构造失败；流式调用不支持工具。
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
    tools: tuple[LLMTool, ...] = Field(default=(), description="模型可以调用的工具")

    @model_validator(mode="after")
    def _reject_tools_with_json_output(self) -> LLMRequestParams:
        if self.tools and self.json_output:
            raise ValueError("tools and json_output cannot be requested together")
        return self


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
    """一次完整调用的结果。

    带工具调用的结果可以没有文本，此时 ``text`` 为空串；提供商既没有返回文本、也没有
    返回工具调用时实现显式失败，不构造本结果。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(description="首个候选的文本内容；只有工具调用时为空串")
    tool_calls: tuple[LLMToolCall, ...] = Field(
        default=(), description="首个候选的工具调用，按提供商返回的顺序"
    )
    assistant_message: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "提供商返回的 assistant 消息原样，调用方继续对话时把它追加进消息；提供商要求"
            "回传的字段（如 DeepSeek 的思考内容）随之保留，调用方不需要知道是哪一家。"
            "实现没有给出时为空"
        ),
    )
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
        messages: list[dict[str, Any]],
        params: LLMRequestParams,
        *,
        stream: Literal[False] = False,
    ) -> Coroutine[Any, Any, LLMCompletion]: ...

    @overload
    def complete(
        self,
        messages: list[dict[str, Any]],
        params: LLMRequestParams,
        *,
        stream: Literal[True],
    ) -> AsyncGenerator[LLMStreamChunk, None]: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        params: LLMRequestParams,
        *,
        stream: bool = False,
    ) -> Coroutine[Any, Any, LLMCompletion] | AsyncGenerator[LLMStreamChunk, None]:
        """``stream=False``（默认）返回可 await 的完整结果，``stream=True`` 返回流。

        消息除 ``role`` 与文本 ``content`` 之外，还可以是完整调用交回的 assistant 消息原样
        与 ``tool`` 消息。流首次拉取时才发出请求；流耗尽、被调用方关闭或在拉取时被取消，
        都会关闭底层响应，关闭失败只记录日志，不替换正在传播的异常或取消。流式调用不支持
        工具，请求带工具时直接拒绝。
        """
        if stream:
            if params.tools:
                raise ValueError("Streaming calls do not support tools.")
            return self._stream(messages, params)
        return self._complete(messages, params)

    @abstractmethod
    async def _complete(
        self,
        messages: list[dict[str, Any]],
        params: LLMRequestParams,
    ) -> LLMCompletion:
        """执行一次完整调用并返回首个候选的文本与工具调用。"""

    @abstractmethod
    def _stream(
        self,
        messages: list[dict[str, Any]],
        params: LLMRequestParams,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """返回流式调用的异步生成器，语义见 :meth:`complete`。"""


async def complete_with_provider_retry(
    llm_service: BaseLLMService, messages: list[dict[str, Any]], params: LLMRequestParams
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
    "LLMTool",
    "LLMToolCall",
    "LLMUsage",
    "complete_with_provider_retry",
    "tool_result_message",
]
