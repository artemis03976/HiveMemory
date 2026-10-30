"""交互载荷、逻辑块与会话轮次测试数据；固定写入时间，不依赖墙上时钟。"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from hivememory.core.models import ActorIdentity, LogicalBlock, TurnEvent, TurnRecord
from hivememory.core.models.interaction import InteractionOutcome
from hivememory.core.protocol.models import (
    ConversationTurn,
    InteractionPayload,
    SessionCompactionCheckpoint,
)


def text_turn_events(user_message: str, reply: str = "") -> list[TurnEvent]:
    """只有问答正文的一轮事件：序号 0 的用户消息，回复非空时其后是一条助手正文。"""
    events = [TurnEvent(kind="user_message", sequence=0, content=user_message)]
    if reply:
        events.append(TurnEvent(kind="assistant_message", sequence=1, content=reply))
    return events


def make_interaction_payload(
    user_message: str = "问题", reply: str = "", **fields: Any
) -> InteractionPayload:
    """用真实交互模型构建一份载荷；没有给出事件时由问答正文写成事件。"""
    fields.setdefault("turn_events", text_turn_events(user_message, reply))
    return InteractionPayload(**fields)


def make_logical_block(
    user_message: str = "问题",
    reply: str = "",
    *,
    turn_events: Sequence[TurnEvent] | None = None,
    identity: ActorIdentity | None = None,
    outcome: InteractionOutcome = "completed",
    **block_fields: Any,
) -> LogicalBlock:
    """用真实模型构建一个逻辑块；没有给出事件时由问答正文写成事件。"""
    events = text_turn_events(user_message, reply) if turn_events is None else turn_events
    return LogicalBlock(
        turn=TurnRecord(
            identity=identity or ActorIdentity(user_id="user-1", agent_id="agent-1"),
            outcome=outcome,
            turn_events=tuple(events),
        ),
        **block_fields,
    )


def make_conversation_turn(
    *,
    agent_id: str = "default",
    interaction_id: str = "history-process",
    user_message: str = "问题",
    reply: str = "",
    **payload_fields: Any,
) -> ConversationTurn:
    """用真实交互模型构建一轮历史。"""
    return ConversationTurn(
        interaction_id=interaction_id,
        agent_id=agent_id,
        topic_id="topic-1",
        created_at=datetime(2026, 10, 15, tzinfo=UTC),
        payload=make_interaction_payload(user_message, reply, **payload_fields),
    )


def make_compaction_checkpoint(**fields) -> SessionCompactionCheckpoint:
    """固定时间的检查点，测试显式给出要覆盖的轮次。"""
    return SessionCompactionCheckpoint(
        **{
            "producer": "alice",
            "format_version": 1,
            "created_at": datetime(2026, 10, 16, tzinfo=UTC),
            **fields,
        }
    )
