"""接力摘要请求预算与输出校验，模型边界使用确定性替身。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from hivememory.agent_runtime.context_budget import estimate_context_tokens
from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
from hivememory.agent_runtime.models import GenerationResult
from hivememory.alice.orchestration.relay_summary import RelaySummarizer, SummaryRejectedError
from hivememory.config.alice import CompactionConfig


@pytest.mark.asyncio
async def test_fold_splits_complete_turns_and_uses_previous_fold_with_same_model():
    """长历史分次请求，后一次继承前次摘要，所有请求都遵守同一输入预算。"""
    generate = AsyncMock(
        side_effect=[GenerationResult(text="first summary"), GenerationResult(text="final summary")]
    )
    options = {"model": "provider/resolved", "max_tokens": 512, "temperature": 0.3}
    model = ResolvedFrameModel(options, 600)
    turns = [
        {"interaction_id": name, "messages": [{"role": "user", "content": name + "x" * 1200}]}
        for name in ("a", "b")
    ]
    result = await RelaySummarizer(SimpleNamespace(generate_async=generate)).fold(
        "old summary",
        turns,
        model=model,
        config=CompactionConfig(summary_max_tokens=64),
        language="en",
    )
    assert result == "final summary"
    requests = [call.args[0] for call in generate.await_args_list]
    assert [json.loads(messages[1]["content"])["previous_summary"] for messages in requests] == [
        "old summary",
        "first summary",
    ]
    assert [
        [turn["interaction_id"] for turn in json.loads(messages[1]["content"])["turns"]]
        for messages in requests
    ] == [["a"], ["b"]]
    assert all(estimate_context_tokens(messages) <= 600 for messages in requests)
    assert [call.kwargs for call in generate.await_args_list] == [
        {"model": "provider/resolved", "max_tokens": 64, "temperature": 0.3}
    ] * 2
    assert options["max_tokens"] == 512


@pytest.mark.asyncio
async def test_batch_boundary_matches_the_estimate_of_the_assembled_request():
    """增量计数与实际发送请求的估算一致：恰好放下时合并，少一个 token 就拆分。"""
    turns = [
        {"interaction_id": "a", "messages": [{"role": "user", "content": "中文资料" * 37 + "x"}]},
        {"interaction_id": "b", "messages": [{"role": "user", "content": "混合 mixed 内容" * 23}]},
    ]

    async def fold(limit):
        generate = AsyncMock(return_value=GenerationResult(text="summary"))
        await RelaySummarizer(SimpleNamespace(generate_async=generate)).fold(
            "旧摘要",
            turns,
            model=ResolvedFrameModel({}, limit),
            config=CompactionConfig(summary_max_tokens=64),
            language="zh",
        )
        return [call.args[0] for call in generate.await_args_list]

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
    "result,reason",
    [
        (GenerationResult(text="  "), "summary_empty"),
        (GenerationResult(text="truncated", finish_reason="length"), "summary_length"),
        (
            GenerationResult(text="filtered", finish_reason="content_filter"),
            "summary_content_filter",
        ),
        (GenerationResult(text="⟪ READ | private"), "summary_contains_mtp"),
        (GenerationResult(text="unexpected ⟫"), "summary_contains_mtp"),
        (GenerationResult(text="x" * 400), "summary_output_too_large"),
    ],
)
async def test_invalid_summary_is_rejected_without_retry(result, reason):
    """空白、截断、过滤、工具语法和超长正文均不能成为检查点摘要。"""
    generate = AsyncMock(return_value=result)
    with pytest.raises(SummaryRejectedError, match=reason):
        await RelaySummarizer(SimpleNamespace(generate_async=generate)).fold(
            "",
            [{"text": "history"}],
            model=ResolvedFrameModel({}, 1000),
            config=CompactionConfig(summary_max_tokens=32),
            language="zh",
        )
    assert generate.await_count == 1


@pytest.mark.asyncio
async def test_oversized_single_turn_is_rejected_before_model_call():
    """不能放下的完整轮次不被任意切开，也不提交必然超限的请求。"""
    generate = AsyncMock()
    with pytest.raises(SummaryRejectedError, match="summary_input_too_large"):
        await RelaySummarizer(SimpleNamespace(generate_async=generate)).fold(
            "",
            [{"text": "x" * 8000}],
            model=ResolvedFrameModel({}, 1000),
            config=CompactionConfig(),
            language="en",
        )
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_failure_is_typed_but_programming_errors_propagate():
    """提供商错误允许回退，错误接线不能被伪装成可用摘要。"""
    generate = AsyncMock(side_effect=litellm.RateLimitError("limited", "provider", "model"))
    summarizer = RelaySummarizer(SimpleNamespace(generate_async=generate))
    with pytest.raises(SummaryRejectedError, match="summary_provider_error"):
        await summarizer.fold(
            "",
            [{"text": "history"}],
            model=ResolvedFrameModel({}, 1000),
            config=CompactionConfig(),
            language="en",
        )
    generate.side_effect = TypeError("bad wiring")
    with pytest.raises(TypeError, match="bad wiring"):
        await summarizer.fold(
            "",
            [{"text": "history"}],
            model=ResolvedFrameModel({}, 1000),
            config=CompactionConfig(),
            language="en",
        )


@pytest.mark.asyncio
async def test_summary_deadline_cancels_worker_and_returns_typed_timeout():
    """超时由调用方截止时间控制，等待中的模型协程被取消并完成清理。"""
    closed = asyncio.Event()

    async def generate(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    with pytest.raises(SummaryRejectedError, match="summary_timeout"):
        await RelaySummarizer(SimpleNamespace(generate_async=generate)).fold(
            "",
            [{"text": "history"}],
            model=ResolvedFrameModel({}, 1000),
            config=CompactionConfig(summary_timeout_seconds=0.001),
            language="en",
        )
    assert closed.is_set()
