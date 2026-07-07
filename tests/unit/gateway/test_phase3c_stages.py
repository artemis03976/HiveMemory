from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.engines.gateway import GatewayEngine
from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentClassificationResult,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    SemanticAnalysisResult,
)
from hivememory.gateway.commands import create_builtin_command_registry
from hivememory.gateway.context import GatewayContextBuilder, SessionContext
from hivememory.gateway.facade import GatewayFacade
from hivememory.gateway.factory import build_gateway_pipeline
from hivememory.gateway.pipeline import GatewayPipeline, GatewayState
from hivememory.gateway.stages import (
    CommandInterceptorStage,
    CompositePlaceholderStage,
    IntentClassifierStage,
)
from hivememory.system.gateway.eye import TheEye


class _NoOpInterceptor:
    def intercept(self, query: str):
        return None


class _CompositeClassifier:
    async def classify(self, message: str, *, gateway_intent=None):
        return IntentClassificationResult(
            intent_type=IntentType.COMPOSITE,
            is_composite=True,
            confidence=0.9,
            reason="测试复合意图",
        )


def _identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


@pytest.mark.asyncio
async def test_facade_process_short_circuits_command_before_hydration():
    provider = MagicMock()
    provider.list_active_topics = AsyncMock(return_value=[])
    facade = GatewayFacade(
        eye=MagicMock(),
        context_builder=GatewayContextBuilder(topic_provider=provider),
        pipeline=GatewayPipeline(),
        command_interceptor=CommandInterceptorStage(create_builtin_command_registry()),
    )

    state = await facade.process("/help", identity=_identity())

    assert state.sealed is True
    assert state.command_result is not None
    assert state.command_result.command_id == "system.help"
    assert state.stage_trace[0].stage_name == "S0.CommandInterceptor"
    assert state.stage_trace[0].short_circuited is True
    provider.list_active_topics.assert_not_awaited()


@pytest.mark.asyncio
async def test_facade_process_runs_phase3c_pipeline_after_hydration():
    provider = MagicMock()
    provider.list_active_topics = AsyncMock(
        return_value=[
            TopicSnapshot(
                topic_id="topic_1",
                topic_title="测试话题",
                state_summary="已有上下文",
            )
        ]
    )
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(
        return_value=SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="重写后的问题",
            search_keywords=["重写", "问题"],
            worth_saving=True,
            reason="测试路由",
            target_topic="topic_1",
        )
    )
    engine = GatewayEngine(
        interceptor=_NoOpInterceptor(),
        semantic_analyzer=analyzer,
    )
    facade = GatewayFacade(
        eye=TheEye(engine=engine),
        context_builder=GatewayContextBuilder(topic_provider=provider),
        pipeline=build_gateway_pipeline(engine),
        command_interceptor=CommandInterceptorStage(create_builtin_command_registry()),
    )

    state = await facade.process("帮我继续这个话题", identity=_identity())

    assert state.sealed is True
    assert state.command_result is None
    assert state.topic_id == "topic_1"
    assert state.rewritten_query == "重写后的问题"
    assert state.search_keywords == ("重写", "问题")
    assert state.memory_write_signal == MemoryWriteSignal.WRITE
    assert state.retrieval_strategy is not None
    assert state.retrieval_strategy.mode == RetrievalMode.HYBRID
    assert [trace.stage_name for trace in state.stage_trace] == [
        "S0.CommandInterceptor",
        "S1.IntentClassifier",
        "S2.CompositePlaceholder",
        "S3.ContextRouter",
        "S4a.MemoryValueJudge",
        "S4b.RetrievalStrategy",
        "S5.PlannerRouter",
    ]
    provider.list_active_topics.assert_awaited_once_with(identity=_identity())
    assert "topic_1: 测试话题" in analyzer.analyze.call_args.kwargs["active_topics_menu"]


@pytest.mark.asyncio
async def test_composite_placeholder_only_records_deferred_signal():
    pipeline = GatewayPipeline(
        [
            IntentClassifierStage(_CompositeClassifier()),
            CompositePlaceholderStage(),
        ]
    )
    state = await pipeline.run(
        "查一下 A，并总结 B",
        SessionContext(identity=_identity()),
    )

    assert state.intent_type == IntentType.COMPOSITE
    assert state.is_composite is True
    assert state.composite_deferred is True
    assert state.composite_deferred_reason == "Phase 3C 暂不执行复合意图分解"


def test_gateway_state_projection_matches_phase2_shape_after_pipeline():
    state = GatewayState(raw_message="原始问题", session_context=SessionContext(identity=_identity()))
    state.intent_type = IntentType.QUERY
    state.topic_id = "topic_1"
    state.rewritten_query = "重写问题"
    state.search_keywords = ["kw"]
    state.memory_write_signal = MemoryWriteSignal.WRITE
    state.seal()

    gaze = state.to_eye_gaze_result()

    assert gaze.intent == GatewayIntent.RAG
    assert gaze.target_topic == "topic_1"
    assert gaze.rewritten_query == "重写问题"
    assert gaze.search_keywords == ["kw"]
    assert gaze.worth_saving is True
