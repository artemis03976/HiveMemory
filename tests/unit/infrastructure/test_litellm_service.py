from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.infrastructure.llm.litellm_service import LiteLLMService
from hivememory.system.config import LLMConfig


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        usage=None,
    )


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


@pytest.mark.asyncio
async def test_acomplete_json_uses_response_format(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _response('{"ok": true}')

    monkeypatch.setattr(
        "hivememory.infrastructure.llm.litellm_service.litellm.acompletion",
        fake_acompletion,
    )

    result = await _service().acomplete_json([{"role": "user", "content": "json"}])

    assert result == '{"ok": true}'
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_acomplete_json_falls_back_when_response_format_fails(monkeypatch):
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

    result = await _service().acomplete_json([{"role": "user", "content": "json"}])

    assert result == '{"ok": true}'
    assert acompletion.await_count == 2
    assert "response_format" in acompletion.await_args_list[0].kwargs
    assert "response_format" not in acompletion.await_args_list[1].kwargs
