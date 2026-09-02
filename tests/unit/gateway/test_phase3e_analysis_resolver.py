"""Gateway Phase 3E 查询分析解析器测试。"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import Identity, LogicalBlock, TopicData, TurnRecord
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
)
from hivememory.engines.gateway.models import QueryUnderstandingResult
from hivememory.engines.gateway.query_understanding import QueryUnderstandingError
from hivememory.gateway.analysis import (
    FallbackUserQueryAnalysisResolver,
    LLMUserQueryAnalysisResolver,
    UserQueryAnalysisContext,
)
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.errors import RecoverableGatewayError
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.config import SystemGatewayConfig, UserQueryAnalysisConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from tests.helpers.workspace import make_identity_scope


def _build_context(raw_message: str, **overrides) -> UserQueryAnalysisContext:
    kwargs = {
        "raw_message": raw_message,
        "identity": Identity(user_id="u1"),
        "candidate_topics": CandidateTopics(),
        "topic_id": "NEW_TOPIC",
    }
    kwargs.update(overrides)
    return UserQueryAnalysisContext(**kwargs)


def _build_topic_data(last_user_query: str = "") -> TopicData:
    now = time.time()
    return TopicData(
        topic_id="topic-1",
        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
        topic_title="三餐推荐",
        blocks=(
            LogicalBlock(
                turn=TurnRecord(
                    identity=Identity(user_id="u1"),
                    user_query=last_user_query,
                    assistant_final_text="好的",
                )
            ),
        )
        if last_user_query
        else (),
        last_update=now,
        last_accessed_at=now,
    )


def _build_engine_mock(
    result: QueryUnderstandingResult | None = None,
    error: Exception | None = None,
) -> AsyncMock:
    engine = AsyncMock()
    if error is not None:
        engine.analyze.side_effect = error
    else:
        engine.analyze.return_value = result or QueryUnderstandingResult(
            intent_type=IntentType.RAG,
            rewritten_query="重写后的查询",
            search_keywords=("关键词",),
            memory_write_signal=MemoryWriteSignal.WRITE,
        )
    return engine


@pytest.mark.asyncio
async def test_fallback_resolver_returns_fixed_conservative_result() -> None:
    resolver = FallbackUserQueryAnalysisResolver(
        UserQueryAnalysisConfig(default_top_k=11)
    )
    context = _build_context("不要改写这个问题")

    result = await resolver.resolve(context)

    assert result.intent_type == IntentType.RAG
    assert result.rewritten_query == "不要改写这个问题"
    assert result.search_keywords == ()
    assert result.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.retrieval_plan.mode == RetrievalMode.HYBRID
    assert result.retrieval_plan.top_k == 11


@pytest.mark.asyncio
async def test_llm_resolver_maps_engine_result() -> None:
    # 差异化 mock 值：若 resolver 映射被清空或硬编码，断言即红
    engine = _build_engine_mock(
        result=QueryUnderstandingResult(
            intent_type=IntentType.RAG,
            rewritten_query="改写后的完全不同",
            search_keywords=("关键词甲", "关键词乙"),
            memory_write_signal=MemoryWriteSignal.WRITE,
        )
    )
    resolver = LLMUserQueryAnalysisResolver(
        config=UserQueryAnalysisConfig(), engine=engine
    )

    result = await resolver.resolve(_build_context("那个报错怎么修"))

    assert result.intent_type == IntentType.RAG
    assert result.rewritten_query == "改写后的完全不同"
    assert result.search_keywords == ("关键词甲", "关键词乙")
    assert result.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.retrieval_plan.mode == RetrievalMode.HYBRID


@pytest.mark.asyncio
async def test_llm_resolver_rule_overrides_explicit_write_intent() -> None:
    engine = _build_engine_mock()
    resolver = LLMUserQueryAnalysisResolver(
        config=UserQueryAnalysisConfig(), engine=engine
    )

    result = await resolver.resolve(_build_context("记住我不吃香菜"))

    assert result.intent_type == IntentType.WRITE
    assert result.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.retrieval_plan.mode == RetrievalMode.SKIP
    assert result.retrieval_plan.top_k == 0


@pytest.mark.asyncio
async def test_llm_resolver_rule_marks_repeated_input_skip() -> None:
    engine = _build_engine_mock()
    resolver = LLMUserQueryAnalysisResolver(
        config=UserQueryAnalysisConfig(), engine=engine
    )
    context = _build_context(
        "晚饭吃什么",
        topic_id="topic-1",
        routed_topic_data=_build_topic_data(last_user_query="晚饭吃什么"),
    )

    result = await resolver.resolve(context)

    assert result.memory_write_signal == MemoryWriteSignal.SKIP
    assert result.intent_type == IntentType.RAG


@pytest.mark.asyncio
async def test_llm_resolver_derives_dense_mode_without_keywords() -> None:
    engine = _build_engine_mock(
        result=QueryUnderstandingResult(
            intent_type=IntentType.RAG,
            rewritten_query="重写后的查询",
            search_keywords=(),
            memory_write_signal=MemoryWriteSignal.UNKNOWN,
        )
    )
    resolver = LLMUserQueryAnalysisResolver(
        config=UserQueryAnalysisConfig(default_top_k=7), engine=engine
    )

    result = await resolver.resolve(_build_context("随便聊聊热点新闻"))

    assert result.retrieval_plan.mode == RetrievalMode.DENSE
    assert result.retrieval_plan.top_k == 7


@pytest.mark.asyncio
async def test_llm_resolver_converts_engine_error_to_recoverable() -> None:
    engine = _build_engine_mock(error=QueryUnderstandingError("解析失败"))
    events = RecordingRuntimeEventSink()
    resolver = LLMUserQueryAnalysisResolver(
        config=UserQueryAnalysisConfig(),
        engine=engine,
        runtime_events=events,
    )

    with pytest.raises(RecoverableGatewayError, match="解析失败"):
        await resolver.resolve(_build_context("查询"))

    capability_events = [
        event
        for event in events.events
        if event.event_type
        == RuntimeEventType.GATEWAY_ANALYSIS_CAPABILITY_COMPLETED.value
    ]
    assert len(capability_events) == 1
    assert capability_events[0].data["error"] == "解析失败"


@pytest.mark.asyncio
async def test_runtime_uses_topic_router_and_llm_resolver() -> None:
    bus = GlobalSystemBus()
    events = RecordingRuntimeEventSink()
    llm_service = AsyncMock()
    llm_service.acomplete_json.side_effect = [
        '{"target_topic":"NEW_TOPIC","new_topic_title":"查询分析",'
        '"new_topic_summary":"验证查询分析解析器","reason":"无匹配话题"}',
        '{"intent_type":"RAG",'
        '"rewritten_query":"标准查询的重写结果",'
        '"search_keywords":["标准查询"],'
        '"memory_write_signal":"WRITE",'
        '"sub_intents":[],"reason":"技术问答"}',
    ]

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
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.rewritten_query == "标准查询的重写结果"
    assert result.decision.search_keywords == ("标准查询",)
    assert result.decision.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.decision.retrieval_plan.mode == RetrievalMode.HYBRID
    assert result.decision.retrieval_plan.top_k == 9
    assert llm_service.acomplete_json.await_count == 2
    completed = [
        event
        for event in events.events
        if event.event_type == RuntimeEventType.GATEWAY_STEP_COMPLETED.value
    ]
    assert completed[-1].data["step_id"] == "user_query_analysis"
    assert completed[-1].data["is_fallback"] is False
    capability_events = [
        event
        for event in events.events
        if event.event_type
        == RuntimeEventType.GATEWAY_ANALYSIS_CAPABILITY_COMPLETED.value
    ]
    assert len(capability_events) == 1
    assert capability_events[0].data["error"] is None


@pytest.mark.asyncio
async def test_runtime_falls_back_to_conservative_result_on_analysis_error() -> None:
    bus = GlobalSystemBus()
    llm_service = AsyncMock()
    llm_service.acomplete_json.side_effect = [
        '{"target_topic":"NEW_TOPIC","new_topic_title":"查询分析",'
        '"new_topic_summary":"验证查询分析兜底","reason":"无匹配话题"}',
        '{"intent_type":"RAG","rewritten_query":"  "}',
    ]

    async def list_active_topics(**_kwargs):
        return ()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, list_active_topics)
    runtime = GatewayRuntime(
        config=SystemGatewayConfig(
            user_query_analysis=UserQueryAnalysisConfig(default_top_k=9)
        ),
        global_bus=bus,
        llm_service=llm_service,
    )

    result = await GatewayService(runtime).process(
        "继续处理标准查询",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.rewritten_query == "继续处理标准查询"
    assert result.decision.memory_write_signal == MemoryWriteSignal.UNKNOWN
    assert result.decision.retrieval_plan.mode == RetrievalMode.HYBRID
    assert result.decision.retrieval_plan.top_k == 9
