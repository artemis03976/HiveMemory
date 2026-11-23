"""LLM 接口模块的带重试完整调用：只对提供商失败重试一次，超窗与程序错误不重试。"""

import pytest

from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.infrastructure.llm.base import LLMRequestParams, complete_with_provider_retry
from tests.helpers.llm import ScriptedLLMService

MESSAGES = [{"role": "user", "content": "Hi"}]
PARAMS = LLMRequestParams(model="test-model")


@pytest.mark.asyncio
async def test_provider_failure_is_retried_once_with_the_same_request():
    llm = ScriptedLLMService(ModelProviderError("transient"), "answer")

    completion = await complete_with_provider_retry(llm, MESSAGES, PARAMS)

    assert completion.text == "answer"
    assert [(call.messages, call.params) for call in llm.calls] == [(MESSAGES, PARAMS)] * 2


@pytest.mark.asyncio
async def test_exhausted_retries_raise_the_last_provider_failure():
    """第二次仍是提供商失败时抛出它，调用方据此按失败收尾，而不是当作没有输出。"""
    last_failure = ModelProviderError("still failing")
    llm = ScriptedLLMService(ModelProviderError("transient"), last_failure)

    with pytest.raises(ModelProviderError) as raised:
        await complete_with_provider_retry(llm, MESSAGES, PARAMS)

    assert raised.value is last_failure
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [ModelContextWindowExceededError("too long"), TypeError("broken prompt wiring")],
    ids=["context-window", "program-error"],
)
async def test_non_retryable_failures_propagate_after_one_attempt(failure):
    """超窗时同一请求必然再次失败，程序错误不属于提供商失败；两者都不重试。"""
    llm = ScriptedLLMService(failure, "answer")

    with pytest.raises(type(failure)) as raised:
        await complete_with_provider_retry(llm, MESSAGES, PARAMS)

    assert raised.value is failure
    assert len(llm.calls) == 1
