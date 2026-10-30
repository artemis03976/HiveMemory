"""
HiveMemory 核心数据模型 - 交互与流转领域

定义用户身份标识和在系统中流转的消息模型。
"""

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from hivememory.core.models.identity import ActorIdentity
from hivememory.core.models.immutable import FrozenDict, freeze_mapping

InteractionOutcome = Literal["completed", "cancelled", "failed"]

# HiveMemory 能力动词：MTP 指令的动词，以及外部 harness 经 HiveMemory MCP 工具
# 调用的能力。外部工具名是开放词表、会与动词重名（Claude Code 有 Read、Write
# 工具），因此动词单独成字段，不与 tool_name 合并。
CapabilityVerb = Literal["SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"]

# 工具调用的结局。
ToolResultStatus = Literal["success", "ack", "error", "cancelled"]

# 每种事件可以携带的可选字段；未列出的字段必须保持缺省（None 或空元组）。
_EVENT_OPTIONAL_FIELDS: dict[str, frozenset[str]] = {
    "user_message": frozenset(),
    "assistant_message": frozenset(),
    "thought": frozenset(),
    "tool_call": frozenset(
        {
            "action_id",
            "tool_name",
            "verb",
            "targets",
            "tool_args",
            "status",
            "result",
            "produced_aliases",
        }
    ),
}
_OPTIONAL_FIELDS = frozenset().union(*_EVENT_OPTIONAL_FIELDS.values())


class StreamMessageType(str, Enum):
    """流式消息类型枚举"""

    USER = "user"  # 用户查询
    SYSTEM = "system"  # 系统消息
    ASSISTANT = "assistant"  # 助手消息
    TOOL = "tool"  # 工具输出
    THOUGHT = "thought"  # 思考过程 (Internal)
    TOOL_CALL = "tool_call"  # 工具调用 (Internal)


class StreamMessage(BaseModel):
    """
    统一流式消息模型

    职责：抹平不同 Agent 框架的消息格式差异，统一系统内的消息流转
    """

    message_type: StreamMessageType
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    # 身份标识
    identity: ActorIdentity = Field(description="身份标识")

    # 工具调用相关字段（可选）
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None

    @property
    def role(self) -> str:
        """映射消息类型到 OpenAI 角色"""
        mapping = {
            StreamMessageType.USER: "user",
            StreamMessageType.ASSISTANT: "assistant",
            StreamMessageType.SYSTEM: "system",
            StreamMessageType.THOUGHT: "assistant",
            StreamMessageType.TOOL_CALL: "assistant",
            StreamMessageType.TOOL: "tool",
        }
        return mapping.get(self.message_type, "assistant")

    model_config = ConfigDict(use_enum_values=True)


class TurnEvent(BaseModel):
    """
    一轮 Agent 交互的原子事件，是一轮交互过程的唯一记录。

    Alice 的 MTP 指令与外部 harness 的工具调用写出同形的事件：字段含义相同，
    来源之间真实存在的差异只体现为取值，读取方不判断来源。

    - ``content`` 是执行者写出的原文：用户消息、助手正文、思考或 MTP 指令
      片段（结构化工具调用没有调用原文，为空串）。原文供重放，对工具调用的
      解读只看结构化字段；MTP 指令解析失败时只有原文。
    - 一次工具调用与它的结局是同一个 ``tool_call`` 事件：调用在执行前写出，
      结局到达时由 ``with_outcome`` 补全。``action_id`` 标识这次调用。
    - ``verb`` 与 ``tool_name`` 互斥，回答“调用了什么”：前者是 HiveMemory
      能力，后者是外部工具的名字；MTP 指令无法解析时二者都为空。
    - ``targets`` 是动词的作用对象：READ 的 alias，RUN 的工具或代码记忆，
      UPDATE 的记忆，CALL 的 Agent；SEARCH 与 WRITE 的通配目标为空。
      ``tool_args`` 只放作用对象以外的参数。
    - ``status`` 与 ``result`` 是结局与执行者收到的返回原文，同时出现或
      同时为空。没有结局表示结局没有记录：执行者拿到结果之前被取消或失败，
      或来源没有记下结果（外部 harness 的会话记录会漏记并行调用的结果），
      因此不等于调用没有完成。
    - ``produced_aliases`` 是本次调用产出的 alias：WRITE、UPDATE 的 ACK 为
      pending alias，CALL 为子帧回填的产物，其他为空。

    单个事件的字段规则在构造时校验；一轮事件集的不变量由
    ``validate_turn_events`` 校验。
    """

    kind: Literal[
        "user_message",
        "assistant_message",
        "thought",
        "tool_call",
    ]
    sequence: int
    content: str

    # ---- 只在 tool_call 上 ----
    # 一次调用的标识，轮内唯一。
    action_id: str | None = None
    tool_name: str | None = None
    verb: CapabilityVerb | None = None
    targets: tuple[str, ...] = ()
    # 运行时保持递归冻结，HTTP/JSON schema 按实际序列化结果声明为普通对象。
    tool_args: (
        Annotated[
            FrozenDict[str, Any], WithJsonSchema({"type": "object", "additionalProperties": True})
        ]
        | None
    ) = None

    # ---- tool_call 的结局 ----
    status: ToolResultStatus | None = None
    result: str | None = None
    produced_aliases: tuple[str, ...] = ()

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("tool_args", mode="before")
    @classmethod
    def _freeze_tool_args(cls, value: Any) -> FrozenDict[str, Any] | None:
        return None if value is None else freeze_mapping(value)

    @model_validator(mode="after")
    def _check_fields_for_kind(self) -> "TurnEvent":
        """字段只出现在允许它的事件种类上；调用必带标识，结局字段成对出现。"""
        present = {
            name
            for name in _OPTIONAL_FIELDS
            if getattr(self, name) is not None and getattr(self, name) != ()
        }
        unexpected = sorted(present - _EVENT_OPTIONAL_FIELDS[self.kind])
        if unexpected:
            raise ValueError(f"{self.kind} event cannot carry: {', '.join(unexpected)}")
        if self.kind == "tool_call" and not self.action_id:
            raise ValueError("tool_call event requires action_id")
        if self.verb is not None and self.tool_name is not None:
            raise ValueError("tool_call event cannot carry both verb and tool_name")
        if self.targets and self.verb is None:
            raise ValueError("tool_call event without verb cannot carry targets")
        if (self.status is None) != (self.result is None):
            raise ValueError("tool_call event carries status and result together")
        if self.produced_aliases and self.status is None:
            raise ValueError("tool_call event without outcome cannot carry produced_aliases")
        return self

    @property
    def has_outcome(self) -> bool:
        """调用是否记录了结局。"""
        return self.status is not None

    def with_outcome(
        self,
        status: ToolResultStatus,
        result: str,
        produced_aliases: Sequence[str] = (),
    ) -> "TurnEvent":
        """补全一次调用的结局，返回替换原事件的新事件，序号不变。

        只对没有结局的调用进行：结局只记录一次，重复补全说明写出方把两次
        返回记到了同一个调用上，显式失败。
        """
        if self.kind != "tool_call":
            raise ValueError(f"{self.kind} event has no outcome")
        if self.has_outcome:
            raise ValueError(f"tool_call already has an outcome: {self.action_id}")
        return TurnEvent.model_validate(
            {
                **self.model_dump(),
                "status": status,
                "result": result,
                "produced_aliases": tuple(produced_aliases),
            }
        )


def validate_turn_events(events: Sequence[TurnEvent]) -> None:
    """校验一轮事件集的不变量，违反时抛出 ``ValueError``，不修补。

    - 首个事件是这一轮唯一的用户消息，序号为 0；序号轮内唯一、严格递增；
    - 每个调用的 ``action_id`` 轮内唯一。
    """
    if not events or events[0].kind != "user_message" or events[0].sequence != 0:
        raise ValueError("turn events must start with the user message at sequence 0")
    previous = 0
    calls: set[str | None] = set()
    for event in events[1:]:
        if event.sequence <= previous:
            raise ValueError("turn event sequences must be strictly increasing")
        previous = event.sequence
        if event.kind == "user_message":
            raise ValueError("a turn has exactly one user message")
        if event.kind == "tool_call":
            if event.action_id in calls:
                raise ValueError(f"duplicate tool_call action_id: {event.action_id}")
            calls.add(event.action_id)


def turn_user_message(events: Sequence[TurnEvent]) -> str:
    """一轮的用户消息：序号 0 的用户消息事件的正文。"""
    for event in events:
        if event.kind == "user_message":
            return event.content
    raise ValueError("turn events have no user message")


def turn_reply_text(events: Sequence[TurnEvent]) -> str:
    """一轮的回复正文：全部助手正文按序号以空行拼接；没有助手正文时为空串。

    两种来源统一按空行拼接；执行者交给交付方的最终回复由执行者自己组装，
    不受这里影响。
    """
    return "\n\n".join(
        event.content
        for event in sorted(events, key=lambda item: item.sequence)
        if event.kind == "assistant_message"
    )


class TurnRecord(BaseModel):
    """
    单轮内容记录：本轮结局、标识、产出它的 actor 与本轮事件集。

    事件集是唯一的内容事实；用户消息、回复正文与过程材料都在读取时从事件
    得到，不在记录中另存一份。
    """

    outcome: InteractionOutcome = "completed"
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    # 产出本轮的 actor；没有可以代替的默认身份，由构造方显式给出。
    identity: ActorIdentity

    turn_events: tuple[TurnEvent, ...] = Field(description="本轮事件集，以序号 0 的用户消息开头")

    model_config = ConfigDict(frozen=True)
