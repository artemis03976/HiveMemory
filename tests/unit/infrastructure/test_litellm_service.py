"""LiteLLMService：请求参数透传、两种调用形态、异常转换与流的关闭；SDK 使用确定性替身。"""

import asyncio
from types import SimpleNamespace

import litellm
import pytest

from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.infrastructure.llm.base import (
    LLMCompletion,
    LLMRequestParams,
    LLMStreamChunk,
    LLMUsage,
)
from hivememory.infrastructure.llm.litellm_service import LiteLLMService


class _RecordingSDK:
    """按顺序返回或抛出预置结果，并记录每次发给 SDK 的参数。"""

    def __init__(self, *results) -> None:
        self.calls: list[dict] = []
        self._results = list(results)

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


_BLOCK = object()


class _FakeStreamResponse:
    """模拟 SDK 的流式响应：逐项产出、抛出或阻塞，并记录关闭次数。"""

    def __init__(self, items, *, close_error: Exception | None = None) -> None:
        self._items = list(items)
        self._close_error = close_error
        self.pull_blocked = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if item is _BLOCK:
            self.pull_blocked.set()
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _completion_response(content, finish_reason="stop", total_tokens=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None,
    )


def _stream_chunk(content=None, finish_reason=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)
        ]
    )


def _install(monkeypatch, sdk: _RecordingSDK) -> None:
    monkeypatch.setattr(
        "hivememory.infrastructure.llm.litellm_service.litellm.acompletion", sdk.acompletion
    )


_MESSAGES = [{"role": "user", "content": "hi"}]
_FULL_PARAMS = LLMRequestParams(
    model="mock-model",
    api_key="test-key",
    api_base="https://example.invalid",
    temperature=0.2,
    top_p=0.9,
    max_tokens=64,
    stop=("⟫",),
)
_FULL_SDK_ARGUMENTS = {
    "model": "mock-model",
    "messages": _MESSAGES,
    "api_key": "test-key",
    "api_base": "https://example.invalid",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 64,
    "stop": ["⟫"],
}


def _provider_error(error_type):
    return error_type(message="private provider detail", model="mock-model", llm_provider="openai")


async def _drain(stream) -> list[LLMStreamChunk]:
    return [chunk async for chunk in stream]


class TestComplete:
    """完整调用：参数逐项透传、响应解析与 JSON 模式。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (_FULL_PARAMS, _FULL_SDK_ARGUMENTS),
            (LLMRequestParams(model="mock-model"), {"model": "mock-model", "messages": _MESSAGES}),
        ],
        ids=["all-params", "model-only"],
    )
    async def test_sdk_receives_exactly_the_params_the_request_carries(
        self, monkeypatch, params, expected
    ):
        """请求携带的参数逐项传给 SDK，未携带的项不由实现补入默认值。"""
        sdk = _RecordingSDK(_completion_response("ok"))
        _install(monkeypatch, sdk)

        await LiteLLMService().complete(_MESSAGES, params)

        assert sdk.calls == [expected]

    @pytest.mark.asyncio
    async def test_returns_text_finish_reason_and_usage(self, monkeypatch):
        _install(
            monkeypatch,
            _RecordingSDK(_completion_response("回复", finish_reason="length", total_tokens=42)),
        )

        completion = await LiteLLMService().complete(_MESSAGES, _FULL_PARAMS)

        assert completion == LLMCompletion(
            text="回复", finish_reason="length", usage=LLMUsage(total_tokens=42)
        )

    @pytest.mark.asyncio
    async def test_missing_text_is_a_provider_failure(self, monkeypatch):
        """提供商没有返回文本时显式失败，不把缺失伪装成空文本。"""
        _install(monkeypatch, _RecordingSDK(_completion_response(None)))

        with pytest.raises(ModelProviderError, match="returned no text"):
            await LiteLLMService().complete(_MESSAGES, _FULL_PARAMS)

    @pytest.mark.asyncio
    async def test_json_output_requests_json_object_format(self, monkeypatch):
        sdk = _RecordingSDK(_completion_response('{"ok": true}'))
        _install(monkeypatch, sdk)

        completion = await LiteLLMService().complete(
            _MESSAGES, LLMRequestParams(model="mock-model", json_output=True)
        )

        assert completion.text == '{"ok": true}'
        assert [call.get("response_format") for call in sdk.calls] == [{"type": "json_object"}]

    @pytest.mark.asyncio
    async def test_json_output_falls_back_only_when_sdk_rejects_the_parameter(self, monkeypatch):
        sdk = _RecordingSDK(
            litellm.UnsupportedParamsError(
                "unsupported response_format", model="mock-model", llm_provider="test"
            ),
            _completion_response('{"ok": true}'),
        )
        _install(monkeypatch, sdk)

        completion = await LiteLLMService().complete(
            _MESSAGES, LLMRequestParams(model="mock-model", json_output=True)
        )

        assert completion.text == '{"ok": true}'
        assert [call.get("response_format") for call in sdk.calls] == [
            {"type": "json_object"},
            None,
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("failure", "expected_type"),
        [
            (TypeError("adapter bug"), TypeError),
            (_provider_error(litellm.ServiceUnavailableError), ModelProviderError),
            (asyncio.CancelledError(), asyncio.CancelledError),
        ],
        ids=["program-error", "provider-error", "cancelled"],
    )
    async def test_json_output_does_not_repeat_unrelated_failures(
        self, monkeypatch, failure, expected_type
    ):
        """只有明确的参数不受支持才允许第二次请求；故障和取消不重发。"""
        sdk = _RecordingSDK(failure)
        _install(monkeypatch, sdk)

        with pytest.raises(expected_type):
            await LiteLLMService().complete(
                _MESSAGES, LLMRequestParams(model="mock-model", json_output=True)
            )

        assert len(sdk.calls) == 1


class TestStream:
    """流式调用：延迟请求、分块语义与底层响应的关闭。"""

    @pytest.mark.asyncio
    async def test_stream_requests_on_first_pull_and_yields_deltas_then_finish_reason(
        self, monkeypatch
    ):
        response = _FakeStreamResponse(
            [
                _stream_chunk("你好"),
                SimpleNamespace(choices=[]),
                _stream_chunk(None),
                _stream_chunk("世界", finish_reason="stop"),
            ]
        )
        sdk = _RecordingSDK(response)
        _install(monkeypatch, sdk)

        stream = LiteLLMService().complete(_MESSAGES, _FULL_PARAMS, stream=True)
        assert sdk.calls == []
        chunks = await _drain(stream)

        assert chunks == [
            LLMStreamChunk(delta="你好"),
            LLMStreamChunk(delta="世界"),
            LLMStreamChunk(finish_reason="stop"),
        ]
        assert sdk.calls == [{**_FULL_SDK_ARGUMENTS, "stream": True}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["exhausted", "closed_by_caller", "cancelled_during_pull"])
    async def test_underlying_response_is_closed_however_the_stream_ends(self, monkeypatch, ending):
        tail = [] if ending == "exhausted" else [_BLOCK]
        response = _FakeStreamResponse([_stream_chunk("a"), *tail])
        _install(monkeypatch, _RecordingSDK(response))
        stream = LiteLLMService().complete(_MESSAGES, _FULL_PARAMS, stream=True)

        if ending == "exhausted":
            await _drain(stream)
        elif ending == "closed_by_caller":
            assert await anext(stream) == LLMStreamChunk(delta="a")
            await stream.aclose()
        else:
            assert await anext(stream) == LLMStreamChunk(delta="a")
            pull = asyncio.create_task(anext(stream))
            await response.pull_blocked.wait()
            pull.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pull

        assert response.close_calls == 1

    @pytest.mark.asyncio
    async def test_close_failure_does_not_replace_cancellation(self, monkeypatch):
        response = _FakeStreamResponse([_BLOCK], close_error=RuntimeError("close failed"))
        _install(monkeypatch, _RecordingSDK(response))
        stream = LiteLLMService().complete(_MESSAGES, _FULL_PARAMS, stream=True)

        pull = asyncio.create_task(anext(stream))
        await response.pull_blocked.wait()
        pull.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pull
        assert response.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["complete", "stream_start", "stream_pull"])
@pytest.mark.parametrize(
    ("provider_type", "error_type", "public_message"),
    [
        (
            litellm.ContextWindowExceededError,
            ModelContextWindowExceededError,
            "Model context window exceeded.",
        ),
        (litellm.RateLimitError, ModelProviderError, "Model provider request failed."),
    ],
)
async def test_provider_errors_are_translated_at_every_boundary(
    monkeypatch, path, provider_type, error_type, public_message
):
    """完整调用、流式启动与拉取分别转换超窗和提供商失败，并保留原因链。"""
    provider_error = _provider_error(provider_type)
    response = _FakeStreamResponse([provider_error])
    sdk = _RecordingSDK(response if path == "stream_pull" else provider_error)
    _install(monkeypatch, sdk)
    service = LiteLLMService()

    with pytest.raises(error_type) as raised:
        if path == "complete":
            await service.complete(_MESSAGES, _FULL_PARAMS)
        else:
            await _drain(service.complete(_MESSAGES, _FULL_PARAMS, stream=True))

    assert type(raised.value) is error_type
    assert str(raised.value) == public_message
    assert raised.value.__cause__ is provider_error
    assert len(sdk.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "failure"),
    [
        ("complete", asyncio.CancelledError()),
        ("stream_start", TypeError("bad wiring")),
        ("stream_pull", TypeError("bad wiring")),
    ],
)
async def test_unclassified_failures_propagate_unchanged(monkeypatch, path, failure):
    """取消与程序错误保持原异常，不能降格成可保守处理的提供商失败。"""
    response = _FakeStreamResponse([failure])
    _install(monkeypatch, _RecordingSDK(response if path == "stream_pull" else failure))
    service = LiteLLMService()

    with pytest.raises(type(failure)) as raised:
        if path == "complete":
            await service.complete(_MESSAGES, _FULL_PARAMS)
        else:
            await _drain(service.complete(_MESSAGES, _FULL_PARAMS, stream=True))

    assert raised.value is failure
