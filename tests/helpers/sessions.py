"""交互载荷、逻辑块、话题与会话轮次测试数据；固定写入时间，不依赖墙上时钟。"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    TopicData,
    TopicProgressEntry,
    TurnEvent,
    TurnRecord,
    WorkspaceIdentity,
)
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


def make_topic_data(
    *,
    workspace_identity: WorkspaceIdentity,
    topic_id: str = "topic-1",
    title: str = "话题",
    turns: Sequence[tuple[str, str] | LogicalBlock] = (),
    progress: Sequence[tuple[str, int]] = (),
) -> TopicData:
    """一个话题：``turns`` 依次是各轮，给出用户消息与回复或构造好的逻辑块；``progress`` 依次是
    进展记录各条的正文与覆盖到第几轮。指回原文的区间按覆盖的轮次写成，经 ``TopicData`` 的校验。"""
    interactions = tuple(
        turn if isinstance(turn, LogicalBlock) else make_logical_block(*turn) for turn in turns
    )
    entries = []
    start = 0
    for text, covered_until in progress:
        first = interactions[start].turn
        last = interactions[covered_until - 1].turn
        entries.append(
            TopicProgressEntry(
                text=text,
                from_turn_id=first.turn_id,
                to_turn_id=last.turn_id,
                to_event_seq=last.turn_events[-1].sequence,
                covered_until=covered_until,
                written_at=1.0,
            )
        )
        start = covered_until
    return TopicData(
        topic_id=topic_id,
        workspace_identity=workspace_identity,
        topic_title=title,
        interactions=interactions,
        last_update=1.0,
        progress_record=tuple(entries),
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
