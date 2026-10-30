"""Import Bus 的输入模型、外部会话键与交互标识推导。

契约要点：
    - 一次导入接收 harness 一轮已经结束的交互的完整记录，不跨请求累积；
      用户消息由服务端写成序号为 0 的事件，其后的事件序号严格递增。回复
      正文不单独上传，由事件中的助手正文得到。
    - 事件与 ``TurnEvent`` 同形，请求校验阶段执行同样的字段规则与一轮的
      不变量，违反时整轮拒绝。``verb`` 标记对 HiveMemory 能力的调用：外部
      harness 只能经 HiveMemory 的 MCP 工具检索与读取，导入因此只接受
      ``SEARCH`` 与 ``READ``，读取必须给出作用对象；其他外部工具只带
      ``tool_name``。导入不认识 MCP 工具的参数结构，参数由连接器归一化。
    - 外部 harness 不产出 alias，事件不接受 ``produced_aliases``。
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

from hivememory.core.models import (
    ActorIdentity,
    TurnEvent,
    freeze_mapping,
    validate_turn_events,
)
from hivememory.core.models.interaction import InteractionOutcome, ToolResultStatus

ImportedEventKind = Literal["assistant_message", "thought", "tool_call"]

# 导入事件可以携带的能力动词：外部 harness 只能经 MCP 检索与读取记忆。
ImportedVerb = Literal["SEARCH", "READ"]

IMPORT_INTERACTION_PREFIX = "import_"


class ImportedTurnEvent(BaseModel):
    """导入轮次中用户消息之后的一个事件。"""

    kind: ImportedEventKind = Field(description="事件种类；用户消息不作为事件导入")
    sequence: int = Field(ge=1, description="轮内序号，从 1 起严格递增；0 留给用户消息")
    content: str = Field(description="事件正文；工具调用为空字符串")
    action_id: str | None = Field(default=None, description="工具调用标识，工具调用必带，轮内唯一")
    tool_name: str | None = Field(
        default=None, description="外部工具的名字；调用 HiveMemory 能力时不设置"
    )
    verb: ImportedVerb | None = Field(
        default=None,
        description=(
            "只为 HiveMemory MCP 工具的调用设置：检索为 SEARCH、读取为 READ，"
            "生成时只保留一行摘要，不再渲染取回的记忆；与 tool_name 互斥"
        ),
    )
    targets: list[str] = Field(
        default_factory=list, description="能力动词的作用对象，READ 为要读取的 alias，不能为空"
    )
    tool_args: dict[str, Any] | None = Field(
        default=None, description="工具调用的参数，不含作用对象"
    )
    status: ToolResultStatus | None = Field(
        default=None,
        description="工具调用的结局：success，出错时为 error；会话记录没有记下结果时不设置",
    )
    result: str | None = Field(
        default=None, description="工具调用的返回正文，与 status 同时设置或同时缺省"
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _require_read_targets(self) -> ImportedTurnEvent:
        """读取必须给出要读的 alias，否则摘要无法说明读了哪些记忆。"""
        if self.verb == "READ" and not self.targets:
            raise ValueError("READ tool_call requires targets")
        return self

    def to_turn_event(self) -> TurnEvent:
        """转换为同形的 ``TurnEvent``；单个事件的字段规则在这里执行。"""
        return TurnEvent(
            kind=self.kind,
            sequence=self.sequence,
            content=self.content,
            action_id=self.action_id,
            tool_name=self.tool_name,
            verb=self.verb,
            targets=tuple(self.targets),
            tool_args=None if self.tool_args is None else freeze_mapping(self.tool_args),
            status=self.status,
            result=self.result,
        )


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
    model_used: str | None = Field(default=None, description="实际使用的模型名，可为空")
    recent_turns: list[ImportedRecentTurn] = Field(
        default_factory=list,
        description="该轮之前最近的若干轮，只供 Gateway 落位，不进入交互载荷",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_turn_events(self) -> ImportedTurn:
        """提前执行封口时的事件规则：序号、调用标识唯一、各种事件可带的字段。

        违反时整轮以 422 拒绝，不等到封口才失败。
        """
        validate_turn_events(self.to_turn_events())
        return self

    @property
    def has_output(self) -> bool:
        """用户消息之后是否有事件；没有输出的轮次与任务进程一样不封口。"""
        return bool(self.events)

    def to_turn_events(self) -> list[TurnEvent]:
        """转换为封口使用的事件集：用户消息写在序号 0，其后是导入的事件。"""
        return [
            TurnEvent(kind="user_message", sequence=0, content=self.user_message),
            *(event.to_turn_event() for event in self.events),
        ]


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
