from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.engines.gateway import (
    ContextRouterEngine,
    IntentClassifierEngine,
    MemoryValueJudgeEngine,
    RetrievalStrategyEngine,
)
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
)


@pytest.mark.asyncio
async def test_intent_classifier_maps_existing_gateway_intent():
    engine = IntentClassifierEngine()

    result = await engine.classify("你好", gateway_intent=GatewayIntent.CHAT)

    assert result.intent_type == IntentType.CHAT
    assert result.is_composite is False


@pytest.mark.asyncio
async def test_context_router_projects_gateway_result():
    gateway_engine = MagicMock()
    gateway_engine.process = AsyncMock(
        return_value=GatewayResult(
            intent=GatewayIntent.RAG,
            rewritten_query="重写后的问题",
            search_keywords=["问题"],
            worth_saving=True,
            reason="测试",
            target_topic="topic_1",
        )
    )

    result = await ContextRouterEngine(gateway_engine).route(
        "原始问题",
        active_topics_menu="topic_1",
    )

    assert result.rewritten_query == "重写后的问题"
    assert result.target_topic == "topic_1"
    assert result.worth_saving is True
    gateway_engine.process.assert_awaited_once_with(
        "原始问题",
        active_topics_menu="topic_1",
    )


@pytest.mark.asyncio
async def test_memory_value_judge_uses_compatible_worth_saving_signal():
    engine = MemoryValueJudgeEngine()

    assert await engine.judge("值得保存", worth_saving=True) == MemoryWriteSignal.WRITE
    assert await engine.judge("闲聊", intent_type=IntentType.CHAT) == MemoryWriteSignal.SKIP


@pytest.mark.asyncio
async def test_retrieval_strategy_defaults_to_hybrid():
    engine = RetrievalStrategyEngine()

    result = await engine.pick(intent_type=IntentType.QUERY, target_topic="topic_1")

    assert result.mode == RetrievalMode.HYBRID
    assert result.top_k == 5
