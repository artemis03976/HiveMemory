"""独立 TopicRouterEngine 测试：候选话题的渲染、输出校验与失败分类。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.config.gateway import TopicRouterConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import TopicProgressEntry, TopicSnapshot
from hivememory.engines.gateway.topic_router import (
    TopicRouterEngine,
    TopicRouterError,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
async def test_topic_router_only_returns_topic_routing_fields() -> None:
    llm = ScriptedLLMService(
        '{"target_topic":"topic-1","new_topic_title":"ignored",'
        '"new_topic_summary":"ignored","reason":"延续现有话题"}'
    )
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    result = await router.route(
        "继续实现",
        topic_snapshots=(
            TopicSnapshot(
                last_update=1_700_000_000.0,
                topic_id="topic-1",
                topic_title="Gateway",
                workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
            ),
        ),
    )

    assert result.topic_id == "topic-1"
    assert result.new_topic_title is None
    assert result.new_topic_summary is None
    assert set(result.model_dump()) == {
        "topic_id",
        "new_topic_title",
        "new_topic_summary",
        "reason",
    }
    # 组件参数原样沿用，只追加 JSON 输出要求。
    assert llm.calls[-1].params == DEFAULT_LLM_PARAMS.model_copy(update={"json_output": True})


def _progress_entry(text: str, covered_until: int) -> TopicProgressEntry:
    return TopicProgressEntry(
        text=text,
        from_turn_id="turn-a",
        to_turn_id="turn-b",
        to_event_seq=1,
        covered_until=covered_until,
        written_at=1_700_000_000.0,
    )


@pytest.mark.asyncio
async def test_candidate_topics_show_the_summary_and_an_excerpt_of_the_latest_progress_entry():
    """候选话题每个一行：标题与话题摘要，写过进展记录时再加最近一条的节选（截到 200 个字符）。"""
    llm = ScriptedLLMService('{"target_topic":"topic-1","reason":"延续"}')
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )
    workspace = make_identity_scope(user_id="u1").workspace_identity

    await router.route(
        "继续",
        topic_snapshots=(
            TopicSnapshot(
                last_update=1_700_000_000.0,
                topic_id="topic-1",
                topic_title="Gateway",
                topic_summary="拆分话题路由",
                workspace_identity=workspace,
                progress_record=(
                    _progress_entry("讨论了拆分方式", 2),
                    _progress_entry("定" * 250, 4),
                ),
            ),
            TopicSnapshot(
                last_update=1_700_000_000.0,
                topic_id="topic-2",
                topic_title="部署",
                topic_summary="发布流程",
                workspace_identity=workspace,
            ),
        ),
    )

    system_prompt = llm.calls[-1].messages[0]["content"]
    assert (
        "- topic-1: Gateway | 拆分话题路由 | " + "定" * 200 + "…\n- topic-2: 部署 | 发布流程\n"
    ) in system_prompt
    assert "讨论了拆分方式" not in system_prompt


@pytest.mark.asyncio
async def test_topic_router_accepts_new_topic_metadata() -> None:
    llm = ScriptedLLMService(
        '{"target_topic":"NEW_TOPIC","new_topic_title":"新问题",'
        '"new_topic_summary":"讨论新的问题","reason":"无候选匹配"}'
    )
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    result = await router.route("新问题", topic_snapshots=())

    assert result.topic_id == "NEW_TOPIC"
    assert result.new_topic_title == "新问题"
    assert result.new_topic_summary == "讨论新的问题"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        '{"target_topic":"not-visible","reason":"非法候选"}',
        '{"target_topic":"NEW_TOPIC","reason":"缺少元数据"}',
        "not-json",
        "[]",
    ],
)
async def test_topic_router_converts_invalid_output_to_recoverable_error(
    payload: str,
) -> None:
    llm = ScriptedLLMService(payload)
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    with pytest.raises(TopicRouterError):
        await router.route("问题", topic_snapshots=())


@pytest.mark.asyncio
async def test_topic_router_reports_missing_capability() -> None:
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=None, llm_params=DEFAULT_LLM_PARAMS
    )

    with pytest.raises(TopicRouterError, match="未配置"):
        await router.route("问题", topic_snapshots=())


@pytest.mark.asyncio
async def test_topic_router_classifies_provider_failure():
    """保留提供商失败作为根因，但能力错误不复制其正文。"""
    failure = ModelProviderError("private")
    llm = ScriptedLLMService(failure)
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )
    with pytest.raises(TopicRouterError, match="提供商调用失败") as caught:
        await router.route("问题", topic_snapshots=())
    assert caught.value.__cause__ is failure
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TypeError("broken adapter"), asyncio.CancelledError()])
async def test_topic_router_propagates_program_errors_and_cancellation(failure):
    """程序缺陷与取消不能伪装为模型格式无效。"""
    llm = ScriptedLLMService(failure)
    router = TopicRouterEngine(
        config=TopicRouterConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )
    with pytest.raises(type(failure)) as caught:
        await router.route("问题", topic_snapshots=())
    assert caught.value is failure
