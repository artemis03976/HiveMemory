"""会话 HTTP 请求/响应模型。

响应只投影前端恢复历史与展示会话状态需要的字段：不含 Workspace 归属，
轮次只保留重放消息块所需的载荷，执行侧的追踪、附件引用与 Gateway 判断
不越过 HTTP 边界。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from hivememory.core.models import TurnEvent
from hivememory.core.models.interaction import InteractionOutcome
from hivememory.core.protocol.models import ConversationTurn, SessionCompactionCheckpoint
from hivememory.workspace.sessions import ConversationSession, SessionSummary


class SessionTitleRequest(BaseModel):
    """创建或重命名会话的标题。"""

    title: str = Field(default="", max_length=200)


class SessionTurnEventResponse(BaseModel):
    """轮次事件的 HTTP 表示；字段与轮次事件相同，工具调用与结果按 ``action_id`` 配对。

    调用一方的字段（``verb``/``tool_name``、``targets``、``tool_args``）只出现在
    调用上，结局一方的字段（``status``、``produced_aliases``）只出现在结果上。
    """

    kind: str
    sequence: int
    content: str
    action_id: str | None = None
    tool_name: str | None = None
    verb: str | None = None
    targets: list[str] = Field(default_factory=list)
    tool_args: dict[str, Any] | None = None
    status: str | None = None
    produced_aliases: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, event: TurnEvent) -> SessionTurnEventResponse:
        return cls(
            kind=event.kind,
            sequence=event.sequence,
            content=event.content,
            action_id=event.action_id,
            tool_name=event.tool_name,
            verb=event.verb,
            targets=list(event.targets),
            tool_args=event.tool_args,
            status=event.status,
            produced_aliases=list(event.produced_aliases),
        )


class SessionTurnPayloadResponse(BaseModel):
    """已封口轮次中用于重放的正文。"""

    outcome: InteractionOutcome
    user_message: str
    assistant_final_text: str | None = None
    turn_events: list[SessionTurnEventResponse]


class SessionTurnResponse(BaseModel):
    """一轮已封口交互的 HTTP 表示；``agent_id`` 只用于历史角色展示。"""

    interaction_id: str
    agent_id: str
    topic_id: str
    created_at: datetime
    payload: SessionTurnPayloadResponse

    @classmethod
    def from_domain(cls, turn: ConversationTurn) -> SessionTurnResponse:
        payload = turn.payload
        return cls(
            interaction_id=turn.interaction_id,
            agent_id=turn.agent_id,
            topic_id=turn.topic_id,
            created_at=turn.created_at,
            payload=SessionTurnPayloadResponse(
                outcome=payload.outcome,
                user_message=payload.user_message,
                assistant_final_text=payload.assistant_final_text,
                turn_events=[
                    SessionTurnEventResponse.from_domain(event) for event in payload.turn_events
                ],
            ),
        )


class SessionCompactionCheckpointResponse(BaseModel):
    """会话接受的历史视图检查点；游标只标记原文范围，原文仍完整返回。"""

    producer: str
    format_version: int
    reduced_through: str | None = None
    summarized_through: str | None = None
    summary: str = ""
    references: list[str]
    created_at: datetime

    @classmethod
    def from_domain(
        cls, checkpoint: SessionCompactionCheckpoint | None
    ) -> SessionCompactionCheckpointResponse | None:
        if checkpoint is None:
            return None
        return cls(
            producer=checkpoint.producer,
            format_version=checkpoint.format_version,
            reduced_through=checkpoint.reduced_through,
            summarized_through=checkpoint.summarized_through,
            summary=checkpoint.summary,
            references=list(checkpoint.references),
            created_at=checkpoint.created_at,
        )


class SessionSummaryResponse(BaseModel):
    """会话列表项；``active_process_id`` 非空表示会话正被任务进程占用。"""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int
    active_process_id: str | None = None

    @classmethod
    def from_domain(cls, summary: SessionSummary) -> SessionSummaryResponse:
        return cls(
            session_id=summary.session_id,
            title=summary.title,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            turn_count=summary.turn_count,
            active_process_id=summary.active_process_id,
        )


class ConversationSessionResponse(SessionSummaryResponse):
    """会话全部轮次与历史视图检查点。"""

    turns: list[SessionTurnResponse]
    compaction_checkpoint: SessionCompactionCheckpointResponse | None = None

    @classmethod
    def from_session(cls, session: ConversationSession) -> ConversationSessionResponse:
        return cls(
            **SessionSummaryResponse.from_domain(session).model_dump(),
            turns=[SessionTurnResponse.from_domain(turn) for turn in session.turns],
            compaction_checkpoint=SessionCompactionCheckpointResponse.from_domain(
                session.compaction_checkpoint
            ),
        )


__all__ = [
    "ConversationSessionResponse",
    "SessionCompactionCheckpointResponse",
    "SessionSummaryResponse",
    "SessionTitleRequest",
    "SessionTurnEventResponse",
    "SessionTurnPayloadResponse",
    "SessionTurnResponse",
]
