"""
HiveMemory 核心数据模型 - 交互与流转领域

定义用户身份标识和在系统中流转的消息模型。
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.constants import DEFAULT_USER_ID, DEFAULT_AGENT_ID, DEFAULT_TEAM_ID
from hivememory.utils.token_estimator import estimate_tokens


class Identity(BaseModel):
    """
    身份标识组合 - 统一管理用户、Agent 两个核心ID

    用于替代散落的 user_id, agent_id 参数，
    提供统一的身份标识和便捷的操作方法。

    注意：session_id 已被移除，其功能被 topic_id 替代。
    话题的生命周期由 PerceptionLayer 的 topic_id 管理。

    Attributes:
        user_id: 用户标识符
        agent_id: Agent 标识符
    """
    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    team_id: Optional[str] = Field(default=DEFAULT_TEAM_ID, description="团队 ID（用于 Workspace 作用域过滤）")
    session_id: Optional[str] = Field(default=None, description="会话 ID（兼容字段）")

    @property
    def buffer_key(self) -> str:
        """生成用于缓冲区的唯一键"""
        return f"{self.user_id}:{self.agent_id}"

    @property
    def is_valid(self) -> bool:
        """检查身份标识是否有效"""
        return bool(self.user_id and self.agent_id)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "agent_id": "chatbot",
                "session_id": "sess_456"
            }
        }
    )


class StreamMessageType(str, Enum):
    """流式消息类型枚举"""
    USER = "user"             # 用户查询
    SYSTEM = "system"         # 系统消息
    ASSISTANT = "assistant"   # 助手消息
    TOOL = "tool"             # 工具输出
    THOUGHT = "thought"       # 思考过程 (Internal)
    TOOL_CALL = "tool_call"   # 工具调用 (Internal)


class StreamMessage(BaseModel):
    """
    统一流式消息模型

    职责：抹平不同 Agent 框架的消息格式差异，统一系统内的消息流转
    """
    message_type: StreamMessageType
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    # 身份标识
    identity: Identity = Field(default_factory=Identity, description="身份标识")

    # 工具调用相关字段（可选）
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None

    @property
    def user_id(self) -> str:
        """获取用户 ID (兼容属性)"""
        return self.identity.user_id

    @property
    def agent_id(self) -> str:
        """获取 Agent ID (兼容属性)"""
        return self.identity.agent_id

    @property
    def session_id(self) -> Optional[str]:
        """获取会话 ID (兼容属性)"""
        return self.identity.session_id

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

    @property
    def token_count(self) -> int:
        """估算消息的 Token 数量"""
        return estimate_tokens(self.content)

    def to_langchain_message(self) -> Dict[str, str]:
        """转换为 LangChain 消息格式"""
        return {
            "role": self.role,
            "content": self.content
        }

    model_config = ConfigDict(use_enum_values=True)


class TurnEvent(BaseModel):
    """
    通用 Agent 交互原子事件

    用于统一描述各类 Agent 系统中的最小交互单元：
    - 普通用户/助手消息
    - Thought 片段
    - Tool Call
    - Tool Result
    - 其他系统级注入消息

    设计目标：
    - 作为历史视图重放的统一事件真相源
    - 通过 action_id 维护一次完整 Agent Action 的结构边界
    - 为主动模式（MTP）与被动模式（JSON function call）提供相同事件语义

    当前仅接受标准化后的事件语义与字段命名。
    """

    kind: Literal[
        "user_message",
        "assistant_message",
        "thought",
        "tool_call",
        "tool_result",
        "system_message",
    ]
    sequence: int
    role: Literal["user", "assistant", "system"]
    content: str

    # 聚合信息：将多个原子事件归并为同一次完整 action
    action_id: Optional[str] = None

    # 通用工具调用元数据
    tool_name: Optional[str] = None
    tool_kind: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    target: Optional[str] = None
    status: Optional[str] = None

    # 历史视图渲染提示
    render_as: Literal["plain", "system_tool_result", "system_ipc_return"] = "plain"

    model_config = ConfigDict(use_enum_values=True)


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
    tool_args: Optional[Dict[str, Any]] = None
    results: list[TurnEvent] = Field(default_factory=list)
    status: Optional[str] = None

    @property
    def is_started(self) -> bool:
        """是否已形成一次可识别的动作。"""
        return bool(self.tool_name or self.tool_kind or self.thought or self.results)

    @property
    def is_complete(self) -> bool:
        """是否已拿到至少一个结果事件。"""
        return bool(self.tool_name or self.tool_kind) and bool(self.results)

    @property
    def has_pending_result(self) -> bool:
        """是否已发起动作但尚未收到结果。"""
        return bool(self.tool_name or self.tool_kind) and not self.results

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


class TraceItem(BaseModel):
    """
    Agent Action 的语义摘要项

    作为 generation / relay 等摘要视图的轻量输入，仅保留动作语义：
    - READ   -> 目标
    - SEARCH -> 查询意图
    - RUN    -> 工具与状态

    注意：
    - WRITE / UPDATE 等控制信号默认不转成 TraceItem
    - tool_result 正文与系统回填文本不进入 TraceItem
    """

    action: str = Field(..., description="操作类型: READ / SEARCH / RUN")
    action_id: Optional[str] = Field(default=None, description="来源 AgentAction 的 action_id")
    target: Optional[str] = Field(default=None, description="READ 目标别名")
    query: Optional[str] = Field(default=None, description="SEARCH 查询文本")
    tool: Optional[str] = Field(default=None, description="RUN 工具名称")
    status: Optional[str] = Field(default=None, description="RUN 执行状态")

    model_config = ConfigDict(use_enum_values=True)


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

    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    identity: Identity = Field(default_factory=Identity)

    user_query: str = Field(default="", description="原始用户问题")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="重写后的查询（指代消解与上下文补全）",
    )
    assistant_final_text: str = Field(
        default="",
        description="本轮最终自然语言回复",
    )

    turn_events: list[TurnEvent] = Field(
        default_factory=list,
        description="本轮原子事件流",
    )
    actions: list[AgentAction] = Field(
        default_factory=list,
        description="由 turn_events 聚合出的完整动作单元",
    )
    semantic_traces: list[TraceItem] = Field(
        default_factory=list,
        description="由 actions 派生出的动作摘要缓存",
    )

    @property
    def anchor_text(self) -> str:
        """获取单轮语义锚点。"""
        return self.rewritten_query or self.user_query or ""

    @property
    def has_structured_content(self) -> bool:
        """是否包含结构化事件/动作/摘要信息。"""
        return bool(self.turn_events or self.actions or self.semantic_traces)

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
    def reduce(cls, turn_events: list[TurnEvent | Dict[str, Any] | Any]) -> list[AgentAction]:
        """将事件流聚合为动作列表，保持首次出现顺序。"""
        actions_by_id: dict[str, AgentAction] = {}
        action_order: list[str] = []
        last_action_id: Optional[str] = None

        normalized_events = [
            cls._normalize_event(event)
            for event in sorted(
                turn_events,
                key=lambda e: e.get("sequence", 0) if isinstance(e, dict) else getattr(e, "sequence", 0),
            )
        ]

        for event in normalized_events:
            if event.kind not in {"thought", "tool_call", "tool_result"}:
                continue

            action_id = cls._resolve_action_id(event, last_action_id)
            if action_id is None:
                continue

            action = actions_by_id.get(action_id)
            if action is None:
                action = AgentAction(action_id=action_id)
                actions_by_id[action_id] = action
                action_order.append(action_id)

            if event.kind == "thought":
                action.thought = cls._merge_thought(action.thought, event.content)
            elif event.kind == "tool_call":
                action.tool_name = event.tool_name or event.target or action.tool_name
                action.tool_kind = event.tool_kind or action.tool_kind
                if event.tool_args is not None:
                    action.tool_args = event.tool_args
                action.status = event.status or action.status
            elif event.kind == "tool_result":
                action.results.append(event)
                action.status = event.status or action.status
                if not action.tool_name:
                    action.tool_name = event.tool_name or event.target or action.tool_name
                if not action.tool_kind:
                    action.tool_kind = event.tool_kind or action.tool_kind

            last_action_id = action_id

        return [actions_by_id[action_id] for action_id in action_order if actions_by_id[action_id].is_started]

    @classmethod
    def _normalize_event(cls, event: TurnEvent | Dict[str, Any] | Any) -> TurnEvent:
        """统一兼容对象与 dict 输入。"""
        if isinstance(event, TurnEvent):
            return event
        if isinstance(event, dict):
            return TurnEvent.model_validate(event)
        return TurnEvent.model_validate(event.model_dump())

    @classmethod
    def _resolve_action_id(cls, event: TurnEvent, last_action_id: Optional[str]) -> Optional[str]:
        """解析事件应归属的 action_id。"""
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
    def reduce(cls, actions: list[AgentAction | Dict[str, Any] | Any]) -> list[TraceItem]:
        """将动作列表转换为摘要轨迹列表。"""
        traces: list[TraceItem] = []
        for action in actions:
            normalized = cls._normalize_action(action)
            trace = cls._action_to_trace(normalized)
            if trace is not None:
                traces.append(trace)
        return traces

    @classmethod
    def _normalize_action(cls, action: AgentAction | Dict[str, Any] | Any) -> AgentAction:
        """统一兼容对象与 dict 输入。"""
        if isinstance(action, AgentAction):
            return action
        if isinstance(action, dict):
            return AgentAction.model_validate(action)
        return AgentAction.model_validate(action.model_dump())

    @classmethod
    def _action_to_trace(cls, action: AgentAction) -> Optional[TraceItem]:
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

        # WRITE / UPDATE / CALL / UNKNOWN -> 默认不进入摘要轨迹
        return None

    @staticmethod
    def _extract_search_query(action: AgentAction) -> Optional[str]:
        """优先从 tool_args 提取 SEARCH query，必要时回退解析 thought。"""
        if action.tool_args:
            query = action.tool_args.get("query")
            if isinstance(query, str):
                return query
        return None
