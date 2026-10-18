"""接力摘要请求预算与输出校验；摘要经真实 MTP 生成器调用，LLM 服务使用确定性替身。"""

import asyncio
import json

import pytest

from hivememory.agent_runtime.model.budget import estimate_context_tokens
from hivememory.agent_runtime.model.generator import MTPGenerator
from hivememory.agent_runtime.model.resolution import ResolvedFrameModel
from hivememory.alice.context.summarizer import RelaySummarizer, SummaryRejectedError
from hivememory.config.alice import CompactionConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.mtp import MTP_STOP_SEQUENCE
from hivememory.infrastructure.llm.base import BaseLLMService, LLMCompletion, LLMRequestParams
from tests.helpers.llm import CallbackLLMService, ScriptedLLMService


def _model(limit: int, **params) -> ResolvedFrameModel:
    return ResolvedFrameModel(LLMRequestParams(model="provider/resolved", **params), limit)


def _summarizer(llm: BaseLLMService) -> RelaySummarizer:
    return RelaySummarizer(MTPGenerator(llm))


async def _summary(_messages, _params) -> str:
    return "summary"


@pytest.mark.asyncio
async def test_fold_splits_complete_turns_and_uses_previous_fold_with_same_model():
    """长历史分次请求，后一次继承前次摘要，所有请求都遵守同一输入预算。"""
    llm = ScriptedLLMService("first summary", "final summary")
    model = _model(600, max_tokens=512, temperature=0.3)
    turns = [
        {"interaction_id": name, "messages": [{"role": "user", "content": name + "x" * 1200}]}
        for name in ("a", "b")
    ]
    result = await _summarizer(llm).fold(
        "old summary",
        turns,
        model=model,
        config=CompactionConfig(summary_max_tokens=64),
        language="en",
    )
    assert result == "final summary"
    requests = [call.messages for call in llm.calls]
    assert [json.loads(messages[1]["content"])["previous_summary"] for messages in requests] == [
        "old summary",
        "first summary",
    ]
    assert [
        [turn["interaction_id"] for turn in json.loads(messages[1]["content"])["turns"]]
        for messages in requests
    ] == [["a"], ["b"]]
    assert all(estimate_context_tokens(messages) <= 600 for messages in requests)
    # 沿用本次解析的模型参数，只把输出上限收紧到摘要预算。
    assert [call.params for call in llm.calls] == [
        model.params.model_copy(update={"max_tokens": 64, "stop": (MTP_STOP_SEQUENCE,)})
    ] * 2


@pytest.mark.asyncio
async def test_batch_boundary_matches_the_estimate_of_the_assembled_request():
    """增量计数与实际发送请求的估算一致：恰好放下时合并，少一个 token 就拆分。"""
    turns = [
        {"interaction_id": "a", "messages": [{"role": "user", "content": "中文资料" * 37 + "x"}]},
        {"interaction_id": "b", "messages": [{"role": "user", "content": "混合 mixed 内容" * 23}]},
    ]

    async def fold(limit):
        llm = CallbackLLMService(complete=_summary)
        await _summarizer(llm).fold(
            "旧摘要",
            turns,
            model=_model(limit),
            config=CompactionConfig(summary_max_tokens=64),
            language="zh",
        )
        return [call.messages for call in llm.calls]

    (combined,) = await fold(100000)
    exact = estimate_context_tokens(combined)
    assert [turn["interaction_id"] for turn in json.loads(combined[1]["content"])["turns"]] == [
        "a",
        "b",
    ]
    assert len(await fold(exact)) == 1
    split = await fold(exact - 1)
    assert [
        [turn["interaction_id"] for turn in json.loads(messages[1]["content"])["turns"]]
        for messages in split
    ] == [["a"], ["b"]]
    assert all(estimate_context_tokens(messages) <= exact - 1 for messages in split)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion,reason",
    [
        (LLMCompletion(text="  ", finish_reason="stop"), "summary_empty"),
        (LLMCompletion(text="truncated", finish_reason="length"), "summary_length"),
        (
            LLMCompletion(text="filtered", finish_reason="content_filter"),
            "summary_content_filter",
        ),
        (LLMCompletion(text="⟪ READ | private", finish_reason="stop"), "summary_contains_mtp"),
        (LLMCompletion(text="unexpected ⟫", finish_reason="stop"), "summary_contains_mtp"),
        (LLMCompletion(text="x" * 400, finish_reason="stop"), "summary_output_too_large"),
    ],
)
async def test_invalid_summary_is_rejected_without_retry(completion, reason):
    """空白、截断、过滤、工具语法和超长正文均不能成为检查点摘要。"""
    llm = ScriptedLLMService(completion)
    with pytest.raises(SummaryRejectedError, match=reason):
        await _summarizer(llm).fold(
            "",
            [{"text": "history"}],
            model=_model(1000),
            config=CompactionConfig(summary_max_tokens=32),
            language="zh",
        )
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_oversized_single_turn_is_rejected_before_model_call():
    """不能放下的完整轮次不被任意切开，也不提交必然超限的请求。"""
    llm = ScriptedLLMService()
    with pytest.raises(SummaryRejectedError, match="summary_input_too_large"):
        await _summarizer(llm).fold(
            "",
            [{"text": "x" * 8000}],
            model=_model(1000),
            config=CompactionConfig(),
            language="en",
        )
    assert llm.calls == []


@pytest.mark.asyncio
async def test_provider_failure_is_typed_but_programming_errors_propagate():
    """提供商错误允许回退，错误接线不能被伪装成可用摘要。"""
    summarizer = _summarizer(
        ScriptedLLMService(
            ModelProviderError("Model provider request failed."), TypeError("bad wiring")
        )
    )
    with pytest.raises(SummaryRejectedError, match="summary_provider_error"):
        await summarizer.fold(
            "",
            [{"text": "history"}],
            model=_model(1000),
            config=CompactionConfig(),
            language="en",
        )
    with pytest.raises(TypeError, match="bad wiring"):
        await summarizer.fold(
            "",
            [{"text": "history"}],
            model=_model(1000),
            config=CompactionConfig(),
            language="en",
        )


@pytest.mark.asyncio
async def test_summary_deadline_cancels_worker_and_returns_typed_timeout():
    """超时由调用方截止时间控制，等待中的模型协程被取消并完成清理。"""
    closed = asyncio.Event()

    async def generate(_messages, _params):
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    with pytest.raises(SummaryRejectedError, match="summary_timeout"):
        await _summarizer(CallbackLLMService(complete=generate)).fold(
            "",
            [{"text": "history"}],
            model=_model(1000),
            config=CompactionConfig(summary_timeout_seconds=0.001),
            language="en",
        )
    assert closed.is_set()


@pytest.mark.asyncio
async def test_summary_task_cancellation_propagates_after_worker_cleanup():
    """调用方停止摘要时保留取消信号，并先关闭等待中的模型调用。"""
    started = asyncio.Event()
    closed = asyncio.Event()

    async def generate(_messages, _params):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    task = asyncio.create_task(
        _summarizer(CallbackLLMService(complete=generate)).fold(
            "",
            [{"text": "history"}],
            model=_model(1000),
            config=CompactionConfig(),
            language="en",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
