import pytest
from dataclasses import FrozenInstanceError

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import (
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
)
from hivememory.gateway.context import SessionContext
from hivememory.gateway.pipeline import GatewayPipeline, GatewayState, StageResult
from hivememory.gateway.stages.s0_command import EntryInterceptorStage


class DummyIntentStage:
    """用于验证 StageResult 提交路径的最小 Stage。"""

    stage_name = "S1.DummyIntent"
    writable_fields = frozenset({"intent_type"})

    async def process(self, state: GatewayState) -> StageResult:
        return StageResult.from_updates({"intent_type": IntentType.CHAT})


def _state() -> GatewayState:
    return GatewayState(
        raw_message="hello",
        session_context=SessionContext(identity=Identity(user_id="u1")),
    )


def test_gateway_state_rejects_direct_field_mutation() -> None:
    state = _state()

    with pytest.raises(AttributeError, match="apply_stage_result"):
        state.intent_type = IntentType.CHAT


def test_apply_stage_result_validates_writable_fields() -> None:
    state = _state()

    with pytest.raises(PermissionError, match="无权写入"):
        state.apply_stage_result(
            stage_name="S4.BadStage",
            result=StageResult.from_updates(
                {"memory_write_signal": MemoryWriteSignal.SKIP}
            ),
            duration_ms=1.0,
            writable_fields=frozenset({"intent_type"}),
        )


def test_s0_does_not_own_later_stage_fields() -> None:
    later_stage_fields = {
        "rewritten_query",
        "search_keywords",
        "memory_write_signal",
        "retrieval_strategy",
    }

    assert EntryInterceptorStage.writable_fields.isdisjoint(later_stage_fields)


def test_simple_chat_fallback_is_derived_without_writing_later_fields() -> None:
    state = _state()
    state.apply_stage_result(
        stage_name="S0.EntryInterceptor",
        result=StageResult.from_updates(
            {"intent_type": IntentType.CHAT},
            flow_end_reason="simple_chat",
        ),
        duration_ms=1.0,
        writable_fields=frozenset({"intent_type"}),
    )

    assert state.rewritten_query is None
    assert state.search_keywords == []
    assert state.memory_write_signal is None
    assert state.retrieval_strategy is None

    decision = state.to_prepare_decision()
    assert decision.rewritten_query == state.raw_message
    assert decision.search_keywords == ()
    assert decision.worth_saving is False
    assert decision.memory_write_signal == MemoryWriteSignal.SKIP
    assert decision.retrieval_strategy is not None
    assert decision.retrieval_strategy.mode == RetrievalMode.SKIP
    assert decision.retrieval_request is None


@pytest.mark.asyncio
async def test_pipeline_commits_stage_result_and_seals_state() -> None:
    state = await GatewayPipeline([DummyIntentStage()]).run_state(_state())

    assert state.intent_type == IntentType.CHAT
    assert state.stage_trace[0].stage_name == "S1.DummyIntent"
    assert state.sealed is True

    with pytest.raises(FrozenInstanceError):
        state.intent_type = None
