from dataclasses import FrozenInstanceError

import pytest

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import (
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalStrategy,
)
from hivememory.gateway.context import SessionContext
from hivememory.gateway.pipeline import GatewayPipeline, GatewayState, ShortCircuit, StageTrace


class _RewriteStage:
    stage_name = "rewrite"

    async def process(self, state: GatewayState) -> GatewayState:
        state.intent_type = IntentType.QUERY
        state.rewritten_query = "重写后的查询"
        state.search_keywords = ["重写", "查询"]
        state.topic_id = "topic_1"
        return state


class _TraceStage:
    stage_name = "custom_trace"

    async def process(self, state: GatewayState) -> GatewayState:
        state.stage_trace.append(StageTrace(stage_name=self.stage_name, duration_ms=1.0))
        return state


class _ShortCircuitStage:
    stage_name = "command"

    async def process(self, state: GatewayState) -> GatewayState:
        state.intent_type = IntentType.CHAT
        raise ShortCircuit(state)


class _ShouldNotRunStage:
    async def process(self, state: GatewayState) -> GatewayState:
        raise AssertionError("short-circuit 后不应继续执行")


def _context() -> SessionContext:
    return SessionContext(identity=Identity(user_id="u1", agent_id="a1"))


@pytest.mark.asyncio
async def test_pipeline_runs_stages_and_seals_state():
    pipeline = GatewayPipeline([_RewriteStage()])

    state = await pipeline.run("原始查询", _context())

    assert state.sealed is True
    assert state.rewritten_query == "重写后的查询"
    assert state.search_keywords == ("重写", "查询")
    assert state.stage_trace[0].stage_name == "rewrite"
    with pytest.raises(FrozenInstanceError):
        state.rewritten_query = "不允许修改"
    with pytest.raises(AttributeError):
        state.search_keywords.append("不允许原地修改")


@pytest.mark.asyncio
async def test_pipeline_preserves_stage_owned_trace():
    pipeline = GatewayPipeline([_TraceStage()])

    state = await pipeline.run("hello", _context())

    assert len(state.stage_trace) == 1
    assert state.stage_trace[0].stage_name == "custom_trace"
    assert state.stage_trace[0].duration_ms == 1.0


@pytest.mark.asyncio
async def test_pipeline_short_circuit_seals_and_stops():
    pipeline = GatewayPipeline([_ShortCircuitStage(), _ShouldNotRunStage()])

    state = await pipeline.run("hello", _context())

    assert state.sealed is True
    assert state.intent_type == IntentType.CHAT
    assert state.stage_trace[0].short_circuited is True


def test_gateway_state_projects_to_eye_gaze_result():
    state = GatewayState(raw_message="原始", session_context=_context())
    state.intent_type = IntentType.QUERY
    state.rewritten_query = "重写"
    state.search_keywords = ["kw"]
    state.topic_id = "topic_1"
    state.new_topic_title = "标题"
    state.new_topic_summary = "摘要"
    state.memory_write_signal = MemoryWriteSignal.WRITE
    state.stage_trace.append(StageTrace(stage_name="s1", duration_ms=2.0))
    state.seal()

    gaze = state.to_eye_gaze_result()

    assert gaze.rewritten_query == "重写"
    assert gaze.search_keywords == ["kw"]
    assert gaze.worth_saving is True
    assert gaze.raw_query == "原始"
    assert gaze.identity.user_id == "u1"
    assert gaze.target_topic == "topic_1"
    assert gaze.new_topic_title == "标题"
    assert gaze.new_topic_summary == "摘要"
    assert gaze.processing_time_ms == 2.0


def test_gateway_state_projects_to_retrieval_request():
    state = GatewayState(raw_message="原始", session_context=_context())
    state.rewritten_query = "重写"
    state.search_keywords = ["kw"]
    state.retrieval_strategy = RetrievalStrategy(mode=RetrievalMode.HYBRID, top_k=8)

    request = state.to_retrieval_request()

    assert request is not None
    assert request.semantic_query == "重写"
    assert request.keywords == ["kw"]
    assert request.identity.user_id == "u1"


def test_gateway_state_skip_retrieval_projection_returns_none():
    state = GatewayState(raw_message="原始", session_context=_context())
    state.retrieval_strategy = RetrievalStrategy(mode=RetrievalMode.SKIP, top_k=0)

    assert state.to_retrieval_request() is None


def test_gateway_state_projects_memory_write_signal_to_payload_worth_saving():
    state = GatewayState(raw_message="原始", session_context=_context())

    state.memory_write_signal = MemoryWriteSignal.WRITE
    assert state.to_interaction_payload_worth_saving() is True

    state.memory_write_signal = MemoryWriteSignal.SKIP
    assert state.to_interaction_payload_worth_saving() is False

    state.memory_write_signal = MemoryWriteSignal.UNKNOWN
    assert state.to_interaction_payload_worth_saving() is None
