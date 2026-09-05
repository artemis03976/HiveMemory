"""
TurnRecord 与 LogicalBlock 收敛测试

覆盖:
- TurnRecord 基本属性
- LogicalBlock 必须显式传入 turn
- block 兼容属性继续可读
"""

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    ActorIdentity,
    AgentAction,
    LogicalBlock,
    TraceItem,
    TurnEvent,
    TurnRecord,
)


def _identity() -> ActorIdentity:
    return ActorIdentity(user_id="u1", agent_id="a1")


def test_turn_record_anchor_text():
    turn = TurnRecord(
        identity=_identity(),
        user_query="原始问题",
        rewritten_query="重写问题",
    )
    assert turn.anchor_text == "重写问题"


def test_logical_block_rejects_flat_turn_fields():
    event = TurnEvent(
        kind="assistant_message",
        sequence=0,
        role="assistant",
        content="hello",
    )
    action = AgentAction(action_id="a1", tool_kind="READ", tool_name="alias_x")
    trace = TraceItem(action="READ", action_id="a1", target="alias_x")

    with pytest.raises(ValidationError):
        LogicalBlock(
            identity=_identity(),
            user_query="问题",
            rewritten_query="重写问题",
            assistant_final_text="回答",
            turn_events=[event],
            actions=[action],
            semantic_traces=[trace],
        )


def test_logical_block_accepts_turn_record_directly():
    turn = TurnRecord(
        identity=_identity(),
        user_query="hello",
        assistant_final_text="world",
    )
    block = LogicalBlock(turn=turn)
    assert block.user_query == "hello"
    assert block.assistant_final_text == "world"
