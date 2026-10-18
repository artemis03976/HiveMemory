"""测试专用的 LLM 服务替身：继承真实接口，按顺序回放预置结果或交给回调，并记录调用。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass

from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
    LLMStreamChunk,
)

# 一项预置结果：文本即一次完整调用的结果，异常在调用时抛出，分块序列供流式调用回放。
type ScriptedResult = str | LLMCompletion | BaseException | Sequence[LLMStreamChunk]


@dataclass(frozen=True)
class LLMCall:
    """替身收到的一次调用。"""

    messages: list[dict[str, str]]
    params: LLMRequestParams
    stream: bool


class ScriptedLLMService(BaseLLMService):
    """按顺序回放预置结果；结果用尽时显式失败，避免测试静默多调用一次。"""

    def __init__(self, *results: ScriptedResult) -> None:
        self._results = list(results)
        self.calls: list[LLMCall] = []

    def _next(self, messages: list[dict[str, str]], params: LLMRequestParams, *, stream: bool):
        self.calls.append(LLMCall(messages=messages, params=params, stream=stream))
        if not self._results:
            raise AssertionError("ScriptedLLMService 没有剩余的预置结果")
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def _complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> LLMCompletion:
        result = self._next(messages, params, stream=False)
        if isinstance(result, str):
            return LLMCompletion(text=result, finish_reason="stop")
        if isinstance(result, LLMCompletion):
            return result
        raise AssertionError("完整调用的预置结果必须是文本或 LLMCompletion")

    async def _stream(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        result = self._next(messages, params, stream=True)
        if isinstance(result, str | LLMCompletion):
            raise AssertionError("流式调用的预置结果必须是分块序列")
        for chunk in result:
            yield chunk


type CompleteHandler = Callable[
    [list[dict[str, str]], LLMRequestParams], Awaitable[str | LLMCompletion]
]
type StreamHandler = Callable[
    [list[dict[str, str]], LLMRequestParams], AsyncIterable[str | LLMStreamChunk]
]


class CallbackLLMService(BaseLLMService):
    """把每次调用交给测试回调，适合按消息内容或轮次决定输出的链路测试。

    流式回调可以只产出文本增量；回调没有给出结束原因时，替身按真实实现补上
    ``finish_reason="stop"`` 的最后一项。未提供的形态被调用时显式失败。
    """

    def __init__(
        self,
        *,
        complete: CompleteHandler | None = None,
        stream: StreamHandler | None = None,
    ) -> None:
        self._complete_handler = complete
        self._stream_handler = stream
        self.calls: list[LLMCall] = []

    async def _complete(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> LLMCompletion:
        self.calls.append(LLMCall(messages=messages, params=params, stream=False))
        if self._complete_handler is None:
            raise AssertionError("CallbackLLMService 未提供完整调用的回调")
        result = await self._complete_handler(messages, params)
        return (
            LLMCompletion(text=result, finish_reason="stop") if isinstance(result, str) else result
        )

    async def _stream(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        self.calls.append(LLMCall(messages=messages, params=params, stream=True))
        if self._stream_handler is None:
            raise AssertionError("CallbackLLMService 未提供流式调用的回调")
        finished = False
        async for item in self._stream_handler(messages, params):
            chunk = LLMStreamChunk(delta=item) if isinstance(item, str) else item
            finished = finished or chunk.finish_reason is not None
            yield chunk
        if not finished:
            yield LLMStreamChunk(finish_reason="stop")


DEFAULT_LLM_PARAMS = LLMRequestParams(model="test-model", temperature=0.0, max_tokens=512)


__all__ = ["CallbackLLMService", "DEFAULT_LLM_PARAMS", "LLMCall", "ScriptedLLMService"]
