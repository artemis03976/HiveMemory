"""Gateway Phase 3E 查询分析解析器测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
)
from hivememory.gateway.analysis import (
    FallbackUserQueryAnalysisResolver,
    UserQueryAnalysisContext,
)
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.config import SystemGatewayConfig, UserQueryAnalysisConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink


@pytest.mark.asyncio
async def test_fallback_resolver_returns_fixed_conservative_result() -> None:
    resolver = FallbackUserQueryAnalysisResolver(
        UserQueryAnalysisConfig(default_top_k=11)
    )
    context = UserQueryAnalysisContext(
        raw_message="不要改写这个问题",
        identity=Identity(user_id="u1"),
        candidate_topics=CandidateTopics(),
        topic_id="NEW_TOPIC",
    )

    result = await resolver.resolve(context)

    assert result.intent_type == IntentType.RAG
    assert result.rewritten_query == "不要改写这个问题"
    assert result.search_keywords == ()
    assert result.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.retrieval_plan.mode == RetrievalMode.HYBRID
    assert result.retrieval_plan.top_k == 11


@pytest.mark.asyncio
async def test_runtime_uses_topic_router_and_normal_fallback_resolver() -> None:
    bus = GlobalSystemBus()
    events = RecordingRuntimeEventSink()
    llm_service = AsyncMock()
    llm_service.acomplete_json.return_value = (
        '{"target_topic":"NEW_TOPIC","new_topic_title":"保守分析",'
        '"new_topic_summary":"验证查询分析兜底策略","reason":"无匹配话题"}'
    )

    async def list_active_topics(**_kwargs):
        return ()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, list_active_topics)
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(
            user_query_analysis=UserQueryAnalysisConfig(default_top_k=9)
        ),
        global_bus=bus,
        runtime_events=events,
        llm_service=llm_service,
    )

    result = await GatewayService(runtime).process(
        "继续处理标准查询",
        identity=Identity(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.rewritten_query == "继续处理标准查询"
    assert result.decision.retrieval_plan.top_k == 9
    assert llm_service.acomplete_json.await_count == 1
    completed = [
        event
        for event in events.events
        if event.event_type == RuntimeEventType.GATEWAY_STEP_COMPLETED.value
    ]
    assert completed[-1].data["step_id"] == "user_query_analysis"
    assert completed[-1].data["is_fallback"] is False
