"""独立 TopicRouterEngine 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import TopicSnapshot
from hivememory.engines.gateway.topic_router import (
    TopicRouterEngine,
    TopicRouterError,
)
from hivememory.system.config import TopicRouterConfig
from tests.helpers.workspace import make_access_context


@pytest.mark.asyncio
async def test_topic_router_only_returns_topic_routing_fields() -> None:
    llm = AsyncMock()
    llm.acomplete_json.return_value = (
        '{"target_topic":"topic-1","new_topic_title":"ignored",'
        '"new_topic_summary":"ignored","reason":"延续现有话题"}'
    )
    router = TopicRouterEngine(config=TopicRouterConfig(), llm_service=llm)

    result = await router.route(
        "继续实现",
        topic_snapshots=(
            TopicSnapshot(
                topic_id="topic-1",
                topic_title="Gateway",
                workspace_identity=make_access_context(user_id="u1").workspace_identity,
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


@pytest.mark.asyncio
async def test_topic_router_accepts_new_topic_metadata() -> None:
    llm = AsyncMock()
    llm.acomplete_json.return_value = (
        '{"target_topic":"NEW_TOPIC","new_topic_title":"新问题",'
        '"new_topic_summary":"讨论新的问题","reason":"无候选匹配"}'
    )
    router = TopicRouterEngine(config=TopicRouterConfig(), llm_service=llm)

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
    llm = AsyncMock()
    llm.acomplete_json.return_value = payload
    router = TopicRouterEngine(config=TopicRouterConfig(), llm_service=llm)

    with pytest.raises(TopicRouterError):
        await router.route("问题", topic_snapshots=())


@pytest.mark.asyncio
async def test_topic_router_reports_missing_capability() -> None:
    router = TopicRouterEngine(config=TopicRouterConfig(), llm_service=None)

    with pytest.raises(TopicRouterError, match="未配置"):
        await router.route("问题", topic_snapshots=())
