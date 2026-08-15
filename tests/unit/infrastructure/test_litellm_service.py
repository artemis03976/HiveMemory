from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.infrastructure.llm.litellm_service import (
    LiteLLMService,
    get_gateway_llm_service,
    get_librarian_llm_service,
)
from hivememory.system.config import LLMConfig


def _response(
    content: str,
    usage: SimpleNamespace | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=usage,
    )


@pytest.fixture(autouse=True)
def _reset_litellm_singleton():
    """单例类变量在测试间隔离，避免工厂测试互相污染。"""
    LiteLLMService._instance = None
    LiteLLMService._initialized = False
    yield
    LiteLLMService._instance = None
    LiteLLMService._initialized = False


def _service() -> LiteLLMService:
    service = LiteLLMService.__new__(LiteLLMService)
    service._initialized = False
    LiteLLMService.__init__(
        service,
        LLMConfig(
            model="mock-model",
            api_key="test-key",
            api_base="https://example.invalid",
            temperature=0.1,
            max_tokens=100,
        ),
    )
    return service


def _config() -> LLMConfig:
    return LLMConfig(
        model="mock-model",
        api_key="test-key",
        api_base="https://example.invalid",
        temperature=0.1,
        max_tokens=100,
    )


class TestComplete:
    def test_returns_content_and_passes_defaults(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return _response("hi")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            fake_completion,
        )

        result = _service().complete([{"role": "user", "content": "hi"}])

        assert result == "hi"
        assert calls[0]["model"] == "mock-model"
        assert calls[0]["api_key"] == "test-key"
        assert calls[0]["api_base"] == "https://example.invalid"
        # None 参数回落为实例默认值
        assert calls[0]["temperature"] == 0.1
        assert calls[0]["max_tokens"] == 100

    def test_overrides_defaults_when_explicit(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return _response("hi")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            fake_completion,
        )

        _service().complete(
            [{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=50,
        )

        assert calls[0]["temperature"] == 0.9
        assert calls[0]["max_tokens"] == 50

    def test_forwards_extra_kwargs(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return _response("hi")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            fake_completion,
        )

        _service().complete([{"role": "user", "content": "hi"}], top_p=0.5)

        assert calls[0]["top_p"] == 0.5

    def test_usage_present_does_not_break(self, monkeypatch):
        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            lambda **kwargs: _response(
                "hi",
                usage=SimpleNamespace(total_tokens=42),
            ),
        )
        assert _service().complete([{"role": "user", "content": "hi"}]) == "hi"

    def test_usage_missing_uses_else_branch(self, monkeypatch):
        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            lambda **kwargs: _response("hi", usage=None),
        )
        assert _service().complete([{"role": "user", "content": "hi"}]) == "hi"


class TestCompleteWithTools:
    def test_without_tools_omits_tool_params(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return _response("plain")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            fake_completion,
        )

        result = _service().complete_with_tools(
            [{"role": "user", "content": "hi"}]
        )

        assert result.choices[0].message.content == "plain"
        assert "tools" not in calls[0]
        assert "tool_choice" not in calls[0]

    def test_with_tools_and_choice(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return _response("with tools")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.completion",
            fake_completion,
        )

        tools = [{"type": "function", "function": {"name": "f"}}]
        tool_choice = {"type": "function", "function": {"name": "f"}}

        _service().complete_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice=tool_choice,
        )

        assert calls[0]["tools"] == tools
        assert calls[0]["tool_choice"] == tool_choice

    def test_usage_branches(self, monkeypatch):
        for usage in (SimpleNamespace(total_tokens=7), None):
            def fake_completion(*args, _usage=usage, **kwargs):
                return _response("r", usage=_usage)

            monkeypatch.setattr(
                "hivememory.infrastructure.llm.litellm_service.litellm.completion",
                fake_completion,
            )
            result = _service().complete_with_tools(
                [{"role": "user", "content": "hi"}]
            )
            assert result.choices[0].message.content == "r"


class TestACompleteWithTools:
    @pytest.mark.asyncio
    async def test_awaits_acompletion_with_params(self, monkeypatch):
        acompletion = AsyncMock(
            return_value=_response(
                "async ok",
                usage=SimpleNamespace(total_tokens=21),
            )
        )
        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            acompletion,
        )

        result = await _service().acomplete_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function"}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )

        assert result.choices[0].message.content == "async ok"
        kwargs = acompletion.await_args.kwargs
        assert kwargs["tools"] == [{"type": "function"}]
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "f"}}
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_no_usage_uses_else_branch(self, monkeypatch):
        acompletion = AsyncMock(return_value=_response("async ok", usage=None))
        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            acompletion,
        )

        result = await _service().acomplete_with_tools(
            [{"role": "user", "content": "hi"}]
        )

        assert result.choices[0].message.content == "async ok"


class TestACompleteJson:
    @pytest.mark.asyncio
    async def test_uses_response_format(self, monkeypatch):
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _response('{"ok": true}')

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            fake_acompletion,
        )

        result = await _service().acomplete_json(
            [{"role": "user", "content": "json"}]
        )

        assert result == '{"ok": true}'
        assert calls[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_falls_back_when_response_format_fails(self, monkeypatch):
        acompletion = AsyncMock(
            side_effect=[
                RuntimeError("unsupported response_format"),
                _response('{"ok": true}'),
            ]
        )
        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            acompletion,
        )

        result = await _service().acomplete_json(
            [{"role": "user", "content": "json"}]
        )

        assert result == '{"ok": true}'
        assert acompletion.await_count == 2
        assert "response_format" in acompletion.await_args_list[0].kwargs
        assert "response_format" not in acompletion.await_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_usage_branch_logs(self, monkeypatch):
        async def fake_acompletion(**kwargs):
            return _response('{"ok": true}', usage=SimpleNamespace(total_tokens=9))

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            fake_acompletion,
        )

        result = await _service().acomplete_json(
            [{"role": "user", "content": "json"}]
        )

        assert result == '{"ok": true}'


class TestCompleteWithRetry:
    def test_returns_on_first_success(self, monkeypatch):
        calls = []

        def fake_complete(*args, **kwargs):
            calls.append(args)
            return "ok"

        monkeypatch.setattr(
            LiteLLMService,
            "complete",
            fake_complete,
        )

        result = _service().complete_with_retry([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert len(calls) == 1

    def test_retries_then_returns_none(self, monkeypatch):
        calls = []

        def failing_complete(*args, **kwargs):
            calls.append(args)
            raise RuntimeError("boom")

        monkeypatch.setattr(
            LiteLLMService,
            "complete",
            failing_complete,
        )

        result = _service().complete_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
        )

        assert result is None
        assert len(calls) == 3

    def test_succeeds_after_retry(self, monkeypatch):
        attempts = {"n": 0}

        def flaky_complete(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return "recovered"

        monkeypatch.setattr(
            LiteLLMService,
            "complete",
            flaky_complete,
        )

        result = _service().complete_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
        )

        assert result == "recovered"


class TestAComplete:
    @pytest.mark.asyncio
    async def test_returns_content_with_defaults(self, monkeypatch):
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _response("async hi")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            fake_acompletion,
        )

        result = await _service().acomplete([{"role": "user", "content": "hi"}])

        assert result == "async hi"
        assert calls[0]["model"] == "mock-model"
        assert calls[0]["temperature"] == 0.1
        assert calls[0]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_overrides_defaults_and_forwards_kwargs(self, monkeypatch):
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _response("async hi")

        monkeypatch.setattr(
            "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
            fake_acompletion,
        )

        await _service().acomplete(
            [{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=64,
            top_p=0.9,
        )

        assert calls[0]["temperature"] == 0.7
        assert calls[0]["max_tokens"] == 64
        assert calls[0]["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_usage_and_no_usage_branches(self, monkeypatch):
        for usage in (SimpleNamespace(total_tokens=5), None):
            async def fake_acompletion(*args, _usage=usage, **kwargs):
                return _response("async hi", usage=_usage)

            monkeypatch.setattr(
                "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
                fake_acompletion,
            )
            result = await _service().acomplete(
                [{"role": "user", "content": "hi"}]
            )
            assert result == "async hi"


class TestACompleteWithRetry:
    @pytest.mark.asyncio
    async def test_returns_on_first_success(self, monkeypatch):
        calls = []

        async def fake_acomplete(*args, **kwargs):
            calls.append(args)
            return "ok"

        monkeypatch.setattr(LiteLLMService, "acomplete", fake_acomplete)

        result = await _service().acomplete_with_retry(
            [{"role": "user", "content": "hi"}]
        )

        assert result == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self, monkeypatch):
        attempts = {"n": 0}

        async def flaky_acomplete(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return "recovered"

        monkeypatch.setattr(LiteLLMService, "acomplete", flaky_acomplete)

        result = await _service().acomplete_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
        )

        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_retries_then_returns_none(self, monkeypatch):
        calls = []

        async def failing_acomplete(*args, **kwargs):
            calls.append(args)
            raise RuntimeError("boom")

        monkeypatch.setattr(LiteLLMService, "acomplete", failing_acomplete)

        result = await _service().acomplete_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=2,
        )

        assert result is None
        assert len(calls) == 2


class TestFactories:
    def test_gateway_service_requires_config(self):
        with pytest.raises(ValueError, match="config is required"):
            get_gateway_llm_service(None)

    def test_gateway_service_returns_instance(self):
        service = get_gateway_llm_service(_config())
        assert isinstance(service, LiteLLMService)
        assert service.model == "mock-model"

    def test_librarian_service_with_config(self):
        service = get_librarian_llm_service(_config())
        assert isinstance(service, LiteLLMService)
        assert service.api_key == "test-key"

    def test_librarian_service_loads_global_config(self, monkeypatch):
        calls = []

        def fake_load():
            calls.append(1)
            return SimpleNamespace(
                get_librarian_llm_config=lambda: _config()
            )

        monkeypatch.setattr(
            "hivememory.system.config.load_app_config",
            fake_load,
        )

        service = get_librarian_llm_service(None)

        assert isinstance(service, LiteLLMService)
        assert calls == [1]
