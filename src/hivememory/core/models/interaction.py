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
from hivememory.utils.token_estimator import estimate_tokens

InteractionOutcome = Literal["completed", "cancelled", "failed"]

# HiveMemory 能力动词：MTP 指令的动词，以及外部 harness 经 HiveMemory MCP 工具
# 调用的能力。外部工具名是开放词表、会与动词重名（Claude Code 有 Read、Write
# 工具），因此动词单独成字段，不与 tool_name 合并。
CapabilityVerb = Literal["SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"]

# 工具调用的结局，只写在结果事件上；调用没有结果事件表示这次调用没有完成。
ToolResultStatus = Literal["success", "ack", "error", "cancelled"]

# 每种事件可以携带的可选字段；未列出的字段必须保持缺省（None 或空元组）。
_EVENT_OPTIONAL_FIELDS: dict[str, frozenset[str]] = {
    "user_message": frozenset(),
    "assistant_message": frozenset(),
    "thought": frozenset(),
    "tool_call": frozenset({"action_id", "tool_name", "verb", "targets", "tool_args"}),
    "tool_result": frozenset({"action_id", "status", "produced_aliases"}),
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

    - ``content`` 是执行者写出或收到的原文：用户消息、助手正文、思考、MTP
      指令片段（结构化工具调用没有调用原文，为空串）或工具返回。原文供重放，
      对工具调用的解读只看结构化字段；MTP 指令解析失败时只有原文。
    - 调用一方的事实只写在 ``tool_call`` 上，结局一方的事实只写在
      ``tool_result`` 上，二者以 ``action_id`` 配对，结果不重复调用的字段。
    - ``verb`` 与 ``tool_name`` 互斥，回答“调用了什么”：前者是 HiveMemory
      能力，后者是外部工具的名字；MTP 指令无法解析时二者都为空。
    - ``targets`` 是动词的作用对象：READ 的 alias，RUN 的工具或代码记忆，
      UPDATE 的记忆，CALL 的 Agent；SEARCH 与 WRITE 的通配目标为空。
      ``tool_args`` 只放作用对象以外的参数。
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
        "tool_result",
    ]
    sequence: int
    content: str

    # 调用与结果的配对标识；思考与正文不带。
    action_id: str | None = None

    # ---- 只在 tool_call 上 ----
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

    # ---- 只在 tool_result 上 ----
    status: ToolResultStatus | None = None
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
        """字段只出现在允许它的事件种类上；工具事件必带配对标识，结果必带状态。"""
        present = {
            name
            for name in _OPTIONAL_FIELDS
            if getattr(self, name) is not None and getattr(self, name) != ()
        }
        unexpected = sorted(present - _EVENT_OPTIONAL_FIELDS[self.kind])
        if unexpected:
            raise ValueError(f"{self.kind} event cannot carry: {', '.join(unexpected)}")
        if self.kind in ("tool_call", "tool_result") and not self.action_id:
            raise ValueError(f"{self.kind} event requires action_id")
        if self.kind == "tool_result" and self.status is None:
            raise ValueError("tool_result event requires status")
        if self.verb is not None and self.tool_name is not None:
            raise ValueError("tool_call event cannot carry both verb and tool_name")
        if self.targets and self.verb is None:
            raise ValueError("tool_call event without verb cannot carry targets")
        return self


def validate_turn_events(events: Sequence[TurnEvent]) -> None:
    """校验一轮事件集的不变量，违反时抛出 ``ValueError``，不修补。

    - 首个事件是这一轮唯一的用户消息，序号为 0；序号轮内唯一、严格递增；
    - 每个调用的 ``action_id`` 轮内唯一；每个结果指向本轮更早出现的调用，
      每个调用至多一个结果。没有结果的调用表示这次调用没有完成。
    """
    if not events or events[0].kind != "user_message" or events[0].sequence != 0:
        raise ValueError("turn events must start with the user message at sequence 0")
    previous = 0
    calls: set[str | None] = set()
    answered: set[str | None] = set()
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
        elif event.kind == "tool_result":
            if event.action_id not in calls:
                raise ValueError(f"tool_result without an earlier tool_call: {event.action_id}")
            if event.action_id in answered:
                raise ValueError(f"tool_call has more than one result: {event.action_id}")
            answered.add(event.action_id)


def turn_reply_text(events: Sequence[TurnEvent]) -> str:
    """一轮的回复正文：全部助手正文按序号以空行拼接；没有助手正文时为空串。"""
    return "\n\n".join(
        event.content
        for event in sorted(events, key=lambda item: item.sequence)
        if event.kind == "assistant_message"
    )


class AgentAction(BaseModel):
    """
    完整的 Agent Action 聚合单元

    位于 TurnEvent 之上，用于将一次完整的 Agent 动作保持为同一个结构：
    - 可选 thought
    - 1 次 tool call
    - 0..N 次 tool result

    设计目标：
    - 继承旧三元组 (thought -> tool call - > tool result) 的“动作完整性”设计哲学
    - 适配多段结果、流式返回、挂起恢复等场景
    - 为 TraceItem 摘要层提供稳定输入
    """

    action_id: str
    thought: str = ""
    tool_name: str = ""
    tool_kind: str = ""
    # 运行时保持递归冻结，HTTP/JSON schema 按实际序列化结果声明为普通对象。
    tool_args: (
        Annotated[
            FrozenDict[str, Any], WithJsonSchema({"type": "object", "additionalProperties": True})
        ]
        | None
    ) = None
    results: tuple[TurnEvent, ...] = Field(default_factory=tuple)
    status: str | None = None

    @property
    def is_started(self) -> bool:
        """是否已形成一次可识别的动作。"""
        return bool(self.tool_name or self.tool_kind or self.thought or self.results)

    @property
    def is_complete(self) -> bool:
        """是否已拿到至少一个结果事件。"""
        return bool(self.tool_name or self.tool_kind) and bool(self.results)

    @property
    def total_tokens(self) -> int:
        """估算动作单元的 Token 数量。"""
        tokens = 0
        if self.thought:
            tokens += estimate_tokens(self.thought)
        if self.tool_name:
            tokens += estimate_tokens(self.tool_name)
        if self.tool_kind:
            tokens += estimate_tokens(self.tool_kind)
        if self.tool_args:
            tokens += estimate_tokens(str(self.tool_args))
        for result in self.results:
            tokens += estimate_tokens(result.content)
        return tokens

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("tool_args", mode="before")
    @classmethod
    def _freeze_tool_args(cls, value: Any) -> FrozenDict[str, Any] | None:
        return None if value is None else freeze_mapping(value)


class TraceItem(BaseModel):
    """
    Agent Action 的语义摘要项

    作为 generation / relay 等摘要视图的轻量输入，仅保留动作语义：
    - READ   -> 目标
    - SEARCH -> 查询意图
    - RUN    -> 工具与状态

    注意：
    - WRITE / UPDATE / CALL 等控制信号会保留为 TraceItem，但摘要渲染层可选择跳过
    - tool_result 正文与系统回填文本不进入 TraceItem
    """

    action: str = Field(..., description="操作类型: READ / SEARCH / RUN")
    action_id: str | None = Field(default=None, description="来源 AgentAction 的 action_id")
    target: str | None = Field(default=None, description="READ 目标别名")
    query: str | None = Field(default=None, description="SEARCH 查询文本")
    tool: str | None = Field(default=None, description="RUN 工具名称")
    status: str | None = Field(default=None, description="RUN 执行状态")

    model_config = ConfigDict(frozen=True, use_enum_values=True)


class TurnRecord(BaseModel):
    """
    单轮内容真相记录

    该模型承载一轮交互中与“内容本身”相关的所有结构化信息：
    - user_query / rewritten_query
    - assistant_final_text
    - turn_events
    - actions
    - semantic_traces

    语义定位：
    - TurnRecord 是内容真相源
    - LogicalBlock 是感知层容器
    """

    outcome: InteractionOutcome = "completed"
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    # 产出本轮的 actor；没有可以代替的默认身份，由构造方显式给出。
    identity: ActorIdentity

    user_query: str = Field(default="", description="原始用户问题")
    rewritten_query: str | None = Field(
        default=None,
        description="重写后的查询（指代消解与上下文补全）",
    )
    assistant_final_text: str = Field(
        default="",
        description="本轮最终自然语言回复",
    )

    turn_events: tuple[TurnEvent, ...] = Field(
        default_factory=tuple,
        description="本轮原子事件流",
    )
    actions: tuple[AgentAction, ...] = Field(
        default_factory=tuple,
        description="由 turn_events 聚合出的完整动作单元",
    )
    semantic_traces: tuple[TraceItem, ...] = Field(
        default_factory=tuple,
        description="由 actions 派生出的动作摘要缓存",
    )

    @property
    def anchor_text(self) -> str:
        """获取单轮语义锚点。"""
        return self.rewritten_query or self.user_query or ""

    @property
    def is_empty(self) -> bool:
        """是否尚未承载任何有效内容。"""
        return not (
            self.user_query
            or self.assistant_final_text
            or self.rewritten_query
            or self.turn_events
            or self.actions
            or self.semantic_traces
        )

    model_config = ConfigDict(frozen=True)


class ActionReducer:
    """
    无状态转换器：将 TurnEvent 列表聚合为 AgentAction 列表。

    聚合规则：
    - `tool_call` 创建或更新一个动作单元
    - `tool_result` 追加到对应动作的结果列表
    - `thought` 尝试归并到对应动作；若缺失显式 action_id，则吸附到最近动作
    - 其他事件（如 user/assistant message）不参与动作聚合
    """

    @classmethod
    def reduce(cls, turn_events: Sequence[TurnEvent | dict[str, Any] | Any]) -> list[AgentAction]:
        """将事件流聚合为动作列表，保持首次出现顺序。"""
        actions_by_id: dict[str, dict[str, Any]] = {}
        action_order: list[str] = []
        last_action_id: str | None = None

        normalized_events = [
            cls._normalize_event(event)
            for event in sorted(
                turn_events,
                key=lambda e: (
                    e.get("sequence", 0) if isinstance(e, dict) else getattr(e, "sequence", 0)
                ),
            )
        ]

        for event in normalized_events:
            if event.kind not in {"thought", "tool_call", "tool_result"}:
                continue

            action_id = cls.resolve_action_id(event, last_action_id)
            if action_id is None:
                continue

            action = actions_by_id.get(action_id)
            if action is None:
                action = {
                    "action_id": action_id,
                    "thought": "",
                    "tool_name": "",
                    "tool_kind": "",
                    "tool_args": None,
                    "results": [],
                    "status": None,
                }
                actions_by_id[action_id] = action
                action_order.append(action_id)

            if event.kind == "thought":
                action["thought"] = cls._merge_thought(action["thought"], event.content)
            elif event.kind == "tool_call":
                # 能力调用没有工具名，摘要层以作用对象代替：READ 的 alias、
                # RUN 的工具、CALL 的 Agent。
                action["tool_name"] = event.tool_name or ", ".join(event.targets)
                action["tool_kind"] = event.verb or ""
                action["tool_args"] = event.tool_args
            elif event.kind == "tool_result":
                action["results"].append(event)
                action["status"] = event.status

            last_action_id = action_id

        actions = [AgentAction(**actions_by_id[action_id]) for action_id in action_order]
        return [action for action in actions if action.is_started]

    @classmethod
    def _normalize_event(cls, event: TurnEvent | dict[str, Any] | Any) -> TurnEvent:
        """统一兼容对象与 dict 输入。"""
        if isinstance(event, TurnEvent):
            return event
        if isinstance(event, dict):
            return TurnEvent.model_validate(event)
        return TurnEvent.model_validate(event.model_dump())

    @classmethod
    def resolve_action_id(cls, event: TurnEvent, last_action_id: str | None) -> str | None:
        """解析事件应归属的 action_id；需要按事件定位动作的视图共用这一条规则。

        显式 ``action_id`` 优先；没有时 ``tool_call`` 以序号生成标识，思考与
        工具结果吸附到 ``last_action_id`` 所指的最近动作。
        """
        if event.action_id:
            return event.action_id
        if event.kind == "tool_call":
            return f"tool_call_{event.sequence}"
        if event.kind in {"thought", "tool_result"}:
            return last_action_id
        return None

    @staticmethod
    def _merge_thought(existing: str, incoming: str) -> str:
        """将多段 thought 合并为单个文本块。"""
        if not existing:
            return incoming
        if not incoming:
            return existing
        return f"{existing}\n{incoming}"


class TraceReducer:
    """
    无状态转换器：将 AgentAction 列表化简为 TraceItem 列表。

    这是摘要层的唯一推荐入口。
    """

    @classmethod
    def reduce(cls, actions: Sequence[AgentAction | dict[str, Any] | Any]) -> list[TraceItem]:
        """将动作列表转换为摘要轨迹列表。"""
        traces: list[TraceItem] = []
        for action in actions:
            normalized = cls._normalize_action(action)
            trace = cls._action_to_trace(normalized)
            if trace is not None:
                traces.append(trace)
        return traces

    @classmethod
    def _normalize_action(cls, action: AgentAction | dict[str, Any] | Any) -> AgentAction:
        """统一兼容对象与 dict 输入。"""
        if isinstance(action, AgentAction):
            return action
        if isinstance(action, dict):
            return AgentAction.model_validate(action)
        return AgentAction.model_validate(action.model_dump())

    @classmethod
    def _action_to_trace(cls, action: AgentAction) -> TraceItem | None:
        """将单个动作转换为摘要项。"""
        verb = (action.tool_kind or "").upper()

        if verb == "READ":
            return TraceItem(
                action="READ",
                action_id=action.action_id,
                target=action.tool_name or None,
            )

        if verb == "SEARCH":
            return TraceItem(
                action="SEARCH",
                action_id=action.action_id,
                query=cls._extract_search_query(action),
            )

        if verb == "RUN":
            return TraceItem(
                action="RUN",
                action_id=action.action_id,
                tool=action.tool_name or None,
                status=action.status or "unknown",
            )

        if verb == "CALL":
            return TraceItem(
                action="CALL",
                action_id=action.action_id,
                target=action.tool_name or None,
                status=action.status or "unknown",
            )

        if verb == "WRITE":
            return TraceItem(
                action="WRITE",
                action_id=action.action_id,
                target=action.tool_name or None,
                status=action.status or "unknown",
            )

        if verb == "UPDATE":
            return TraceItem(
                action="UPDATE",
                action_id=action.action_id,
                target=action.tool_name or None,
                status=action.status or "unknown",
            )

        # UNKNOWN -> 默认不进入摘要轨迹
        return None

    @staticmethod
    def _extract_search_query(action: AgentAction) -> str | None:
        """优先从 tool_args 提取 SEARCH query，必要时回退解析 thought。"""
        if action.tool_args:
            query = action.tool_args.get("query")
            if isinstance(query, str):
                return query
        return None
