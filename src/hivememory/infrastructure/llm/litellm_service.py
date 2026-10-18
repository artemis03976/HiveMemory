"""
HiveMemory LiteLLM 服务实现

``BaseLLMService`` 的唯一实现：SDK 参数构造、响应解析、流的迭代与关闭、
异常转换与 JSON 模式回退。服务无状态，不读取配置、不持有默认模型，可被
全部使用方共享。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any

import litellm

from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
    LLMStreamChunk,
    LLMUsage,
)
from hivememory.infrastructure.llm.errors import LLM_PROVIDER_ERRORS

logger = logging.getLogger(__name__)

_JSON_RESPONSE_FORMAT = {"type": "json_object"}


@contextmanager
def _translate_provider_errors() -> Iterator[None]:
    """把 SDK 的超窗与提供商失败转换为中立错误；其他异常与取消原样传播。

    超窗在 SDK 中是请求错误的子类，必须先于提供商失败判断。
    """
    try:
        yield
    except litellm.ContextWindowExceededError as error:
        raise ModelContextWindowExceededError("Model context window exceeded.") from error
    except LLM_PROVIDER_ERRORS as error:
        raise ModelProviderError("Model provider request failed.") from error


def _sdk_arguments(messages: list[dict[str, str]], params: LLMRequestParams) -> dict[str, Any]:
    """按请求参数逐项构造 SDK 参数，不补入请求未携带的项。"""
    arguments: dict[str, Any] = {"model": params.model, "messages": messages}
    optional = {
        "api_key": params.api_key,
        "api_base": params.api_base,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }
    arguments.update({key: value for key, value in optional.items() if value is not None})
    if params.stop:
        arguments["stop"] = list(params.stop)
    return arguments


async def _request(arguments: dict[str, Any], *, json_output: bool) -> Any:
    """发出一次 SDK 请求；JSON 模式只在 SDK 明确不支持该参数时退回普通补全。"""
    if json_output:
        try:
            return await litellm.acompletion(**arguments, response_format=_JSON_RESPONSE_FORMAT)
        except litellm.UnsupportedParamsError:
            logger.warning("LLM JSON 参数不受支持，回退为提示词 JSON: model=%s", arguments["model"])
    return await litellm.acompletion(**arguments)


def _usage(response: Any) -> LLMUsage | None:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    return LLMUsage(total_tokens=total_tokens) if isinstance(total_tokens, int) else None


async def _close_response(response: Any) -> None:
    """关闭底层流式响应；关闭失败只记录日志，不替换正在传播的异常或取消。"""
    close = getattr(response, "aclose", None)
    if not callable(close):
        return
    try:
        close_result = close()
        if inspect.isawaitable(close_result):
            await close_result
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("关闭 LLM 流式响应失败", exc_info=True)


class LiteLLMService(BaseLLMService):
    """
    LiteLLM 服务实现

    使用 litellm 库统一调用各种 LLM 提供商；组合根创建的同一实例由各使用方
    共享，每次调用的模型与采样参数都来自请求参数。
    """

    async def _complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> LLMCompletion:
        """执行一次完整调用；提供商没有返回文本时以提供商失败结束。"""
        logger.debug("调用 LLM: %s, 消息数: %s", params.model, len(messages))
        with _translate_provider_errors():
            response = await _request(
                _sdk_arguments(messages, params), json_output=params.json_output
            )

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice is not None else None
        if choice is None or not isinstance(content, str):
            logger.warning("LLM 响应缺少文本内容 (model=%s)", params.model)
            raise ModelProviderError("Model provider returned no text.")

        completion = LLMCompletion(
            text=content,
            finish_reason=choice.finish_reason or None,
            usage=_usage(response),
        )
        logger.info(
            "LLM 调用成功 (model=%s, tokens=%s, finish_reason=%s)",
            params.model,
            completion.usage.total_tokens if completion.usage else None,
            completion.finish_reason,
        )
        return completion

    async def _stream(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """流式调用；建立请求与每次拉取采用与完整调用相同的异常转换。"""
        arguments = {**_sdk_arguments(messages, params), "stream": True}
        with _translate_provider_errors():
            response = await _request(arguments, json_output=params.json_output)

        finish_reason: str | None = None
        try:
            chunks = response.__aiter__()
            while True:
                try:
                    with _translate_provider_errors():
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta.content or ""
                if delta:
                    yield LLMStreamChunk(delta=delta)
            yield LLMStreamChunk(finish_reason=finish_reason)
        finally:
            await _close_response(response)


__all__ = ["LiteLLMService"]
