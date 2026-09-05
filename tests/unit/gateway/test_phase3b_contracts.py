"""Gateway Phase 3B 公共契约与私有状态测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    ActorIdentity,
    FrozenDict,
    LogicalBlock,
    TopicData,
    TopicLastTurn,
    TopicSnapshot,
    TurnEvent,
    TurnRecord,
)
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayDecision,
    GatewayDecisionOutcome,
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.gateway.analysis import (
    UserQueryAnalysisContext,
)
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.workflow import (
    ExecutionStateStatus,
    GatewayExecutionState,
    GatewayStepResult,
)
from tests.helpers.workspace import make_identity_scope


def test_canonical_topic_object_graph_is_recursively_immutable() -> None:
    event = TurnEvent(
        kind="tool_call",
        sequence=1,
        role="assistant",
        content="查询",
        tool_args={"filters": {"tags": ["gateway"]}},
    )
    block = LogicalBlock(
        turn=TurnRecord(user_query="继续", turn_events=(event,)),
    )
    topic = TopicData(
        topic_id="topic-1",
        workspace_identity=make_identity_scope(user_id="user-1").workspace_identity,
        topic_title="Gateway",
        blocks=[block],
        last_update=1.0,
    )

    assert isinstance(topic.blocks, tuple)
    assert isinstance(event.tool_args, FrozenDict)
    assert event.tool_args["filters"]["tags"] == ("gateway",)
    assert topic.recent_blocks(1) == (block,)

    with pytest.raises(ValidationError):
        topic.topic_title = "被修改"
    with pytest.raises(TypeError):
        event.tool_args["filters"]["tags"] += ("new",)


def test_topic_snapshot_last_turn_and_identity_are_frozen() -> None:
    identity = ActorIdentity(user_id="user-1")
    snapshot = TopicSnapshot(
        topic_id="topic-1",
        topic_title="Gateway",
        workspace_identity=make_identity_scope(user_id="user-1").workspace_identity,
        last_turn=TopicLastTurn(user="问题", assistant="回答"),
    )

    with pytest.raises(ValidationError):
        identity.user_id = "other"
    with pytest.raises(ValidationError):
        snapshot.last_turn.user = "被修改"


def test_public_gateway_result_is_immutable_and_serializable() -> None:
    decision = GatewayDecision(
        target_topic_id="NEW_TOPIC",
        rewritten_query="原问题",
        search_keywords=["gateway"],
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(mode=RetrievalMode.HYBRID, top_k=8),
        intent_type=IntentType.RAG,
    )
    outcome = GatewayDecisionOutcome(decision=decision)
    command = CommandExecutionResult(
        command_id="system.clear",
        status=CommandExecutionStatus.COMPLETED,
        message="完成",
        data={"nested": {"items": [1, 2]}},
    )

    assert outcome.model_dump(mode="json")["kind"] == "decision"
    assert outcome.model_dump(mode="json")["decision"]["search_keywords"] == [
        "gateway"
    ]
    assert command.data["nested"]["items"] == (1, 2)
    with pytest.raises(ValidationError):
        decision.rewritten_query = "被修改"
    with pytest.raises(TypeError):
        command.data["nested"]["items"] += (3,)


def test_private_context_contracts_do_not_duplicate_identity() -> None:
    candidates = CandidateTopics(
        topic_snapshots=[
            TopicSnapshot(
                topic_id="topic-1",
                topic_title="Gateway",
                workspace_identity=make_identity_scope(user_id="user-1").workspace_identity,
            )
        ],
        active_topics_menu="topic-1: Gateway",
    )
    context = UserQueryAnalysisContext(
        raw_message="原问题",
        identity=ActorIdentity(user_id="user-1"),
        candidate_topics=candidates,
        topic_id="topic-1",
    )

    assert isinstance(candidates.topic_snapshots, tuple)
    assert "identity" not in CandidateTopics.model_fields
    assert context.candidate_topics is candidates


def test_execution_state_has_one_guarded_write_entry() -> None:
    state = GatewayExecutionState(
        raw_message="原问题",
        identity_scope=make_identity_scope(user_id="user-1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )
    updates = {"topic_id": "topic-1"}
    result = GatewayStepResult(updates=updates)
    updates["topic_id"] = "被修改"

    state._apply_step_result(result)
    assert state.topic_id == "topic-1"
    assert state.snapshot().topic_id == "topic-1"

    with pytest.raises(ValueError, match="初始化字段"):
        state._apply_step_result(GatewayStepResult(updates={"raw_message": "x"}))
    with pytest.raises(ValueError, match="未知字段"):
        state._apply_step_result(GatewayStepResult(updates={"debug": True}))

    state._apply_step_result(GatewayStepResult(flow_end_reason="command"))
    with pytest.raises(RuntimeError, match="flow end reason"):
        state._apply_step_result(
            GatewayStepResult(
                updates={"topic_id": "不应提交"},
                flow_end_reason="duplicate",
            )
        )
    assert state.topic_id == "topic-1"

    state._mark_completed()
    assert state.status == ExecutionStateStatus.COMPLETED
    with pytest.raises(RuntimeError, match="已完成"):
        state._apply_step_result(GatewayStepResult(updates={"topic_id": "x"}))

    with pytest.raises(FrozenInstanceError):
        state.snapshot().topic_id = "x"
