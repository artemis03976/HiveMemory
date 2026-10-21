"""Import Bus 的输入模型、外部会话键与交互标识推导。

契约要点：
    - 一次导入接收 harness 一轮已经结束的交互的完整记录，不跨请求累积；
      用户消息由服务端写成序号为 0 的事件，其后的事件序号严格递增。
    - 事件不携带 ``role``、``tool_kind``、``target`` 与 ``render_as``：``role``
      由服务端按事件种类填写；``tool_kind`` 只表示 MTP 动词，外部工具不设置，
      因此不会产生 MTP 轨迹，也不会被误当作记忆操作。
    - ``recent_turns`` 只供 Gateway 落位，不进入交互载荷；最近一轮带外部标识
      时，服务端查出它实际落位的话题，供话题路由失败时续接。
    - 交互标识由认证结果与外部标识推导，同一轮重发时不变，不同调用来源、
      actor 或外部来源的同名轮次不会撞号。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.models import ActorIdentity, TurnEvent, freeze_mapping
from hivememory.core.models.interaction import InteractionOutcome

ImportedEventKind = Literal["assistant_message", "thought", "tool_call", "tool_result"]

# 事件种类到 TurnEvent 角色的唯一映射；工具结果是外部系统的回填，记为 system。
_ROLE_BY_KIND: dict[str, Literal["user", "assistant", "system"]] = {
    "assistant_message": "assistant",
    "thought": "assistant",
    "tool_call": "assistant",
    "tool_result": "system",
}

IMPORT_INTERACTION_PREFIX = "import_"


class ImportedTurnEvent(BaseModel):
    """导入轮次中用户消息之后的一个事件。"""

    kind: ImportedEventKind = Field(description="事件种类；用户消息不作为事件导入")
    sequence: int = Field(ge=1, description="轮内序号，从 1 起严格递增；0 留给用户消息")
    content: str = Field(description="事件正文；工具调用可为空字符串")
    action_id: str | None = Field(
        default=None, description="工具调用标识，工具调用与其结果以此配对"
    )
    tool_name: str | None = Field(default=None, description="工具名")
    tool_args: dict[str, Any] | None = Field(default=None, description="工具调用的参数")
    status: str | None = Field(default=None, description="工具结果状态，出错时为 error")

    model_config = ConfigDict(extra="forbid", frozen=True)


class ImportedRecentTurn(BaseModel):
    """该轮之前的一轮，只供 Gateway 理解上文与落位。"""

    user_message: str = Field(description="该轮的用户消息")
    assistant_text: str = Field(default="", description="该轮最终的回复正文")
    external_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="该轮在 harness 中的标识；最近一轮带上它时，服务端查出它的落位，供话题路由失败时续接",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class ImportedTurn(BaseModel):
    """harness 一轮已结束交互的完整记录。"""

    source: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9-]+$",
        description="来源标识，小写字母、数字与连字符，例如 claude-code",
    )
    external_conversation_id: str = Field(
        min_length=1, max_length=256, description="harness 的会话标识"
    )
    external_turn_id: str = Field(
        min_length=1, max_length=256, description="该轮在 harness 中的标识，同一轮重发时不变"
    )
    outcome: InteractionOutcome = Field(description="该轮的结局")
    user_message: str = Field(description="该轮的用户消息")
    events: list[ImportedTurnEvent] = Field(
        default_factory=list, description="用户消息之后的轮次事件，按 sequence 严格递增"
    )
    assistant_final_text: str | None = Field(
        default=None, description="该轮最终的自然语言回复，可为空"
    )
    model_used: str | None = Field(default=None, description="实际使用的模型名，可为空")
    recent_turns: list[ImportedRecentTurn] = Field(
        default_factory=list,
        description="该轮之前最近的若干轮，只供 Gateway 落位，不进入交互载荷",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _require_increasing_sequences(self) -> ImportedTurn:
        """序号决定归约顺序；重复或回退的序号会让动作与回复错位，直接拒绝。"""
        previous = 0
        for event in self.events:
            if event.sequence <= previous:
                raise ValueError("events.sequence must be strictly increasing")
            previous = event.sequence
        return self

    @property
    def has_output(self) -> bool:
        """是否有用户消息之外的输出；没有输出的轮次与任务进程一样不封口。"""
        return bool(self.events) or bool(self.assistant_final_text)

    def to_turn_events(self) -> list[TurnEvent]:
        """转换为感知使用的结构化事件：用户消息写在序号 0，角色按种类填写。"""
        events = [
            TurnEvent(kind="user_message", sequence=0, role="user", content=self.user_message)
        ]
        events.extend(
            TurnEvent(
                kind=event.kind,
                sequence=event.sequence,
                role=_ROLE_BY_KIND[event.kind],
                content=event.content,
                action_id=event.action_id,
                tool_name=event.tool_name,
                tool_args=None if event.tool_args is None else freeze_mapping(event.tool_args),
                status=event.status,
            )
            for event in self.events
        )
        return events


@dataclass(frozen=True)
class ImportConversationKey:
    """同一 actor 在同一外部会话内的串行键。

    外部来源与外部会话标识构成命名空间，再叠加认证通过的 actor，代表不同
    用户或 Agent 的连接器即使会话标识相同也互不阻塞。键只用于进程内串行，
    不参与授权。
    """

    user_id: str
    agent_id: str
    source: str
    external_conversation_id: str

    @classmethod
    def for_turn(cls, actor: ActorIdentity, turn: ImportedTurn) -> ImportConversationKey:
        return cls(
            user_id=actor.user_id,
            agent_id=actor.agent_id,
            source=turn.source,
            external_conversation_id=turn.external_conversation_id,
        )


ImportTurnStatus = Literal["accepted", "duplicate", "skipped"]


@dataclass(frozen=True)
class ImportTurnResult:
    """一次导入的结果：接纳、重复或因没有输出而跳过。"""

    status: ImportTurnStatus
    interaction_id: str


def derive_import_interaction_id(
    *,
    principal_id: str,
    actor: ActorIdentity,
    source: str,
    external_conversation_id: str,
    external_turn_id: str,
) -> str:
    """由认证结果与外部标识推导稳定的导入交互标识。

    对（principal、user、Agent、来源、外部会话、外部轮次）的规范化序列取
    SHA-256 的前 32 位十六进制：同一轮重发得到同一标识，队列据此幂等；
    纳入认证通过的 actor，使代表不同用户或 Agent 的连接器不会撞号；前缀
    把它与任务进程的 ``process_id`` 及其他来源的标识分开。
    """
    canonical = json.dumps(
        [
            principal_id,
            actor.user_id,
            actor.agent_id,
            source,
            external_conversation_id,
            external_turn_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{IMPORT_INTERACTION_PREFIX}{digest[:32]}"


__all__ = [
    "IMPORT_INTERACTION_PREFIX",
    "ImportConversationKey",
    "ImportTurnResult",
    "ImportTurnStatus",
    "ImportedEventKind",
    "ImportedRecentTurn",
    "ImportedTurn",
    "ImportedTurnEvent",
    "derive_import_interaction_id",
]
