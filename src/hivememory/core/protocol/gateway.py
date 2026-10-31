"""Gateway 跨子系统公共协议。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hivememory.core.models import FrozenDict, freeze_mapping


class GatewaySessionTurn(BaseModel):
    """Gateway 分析专用的会话轮次；不携带会话 ID 或授权身份。

    ``topic_id`` 是该轮实际落位的话题，只在话题路由失败时用于续接上一轮：
    同一会话的相邻轮次大多仍在同一话题上，续接比新建无标题话题归错的
    代价小。不知道落位时为空，模型提示词不渲染它。
    """

    user_message: str
    assistant_text: str = ""
    topic_id: str | None = None

    model_config = ConfigDict(frozen=True)


class GatewayIngressMode(str, Enum):
    """Gateway 固定入口策略。"""

    ACTIVE_CHAT = "active_chat"
    PASSIVE_MEMORY = "passive_memory"


class IntentType(str, Enum):
    """Gateway 面向下游的主意图。"""

    RAG = "RAG"
    WRITE = "WRITE"
    CHAT = "CHAT"
    COMPOSITE = "COMPOSITE"
    UNKNOWN = "UNKNOWN"


class RetrievalMode(str, Enum):
    """Gateway 检索模式决策。"""

    DENSE = "DENSE"
    SPARSE = "SPARSE"
    HYBRID = "HYBRID"
    SKIP = "SKIP"


class RetrievalPlan(BaseModel):
    """Gateway 的依赖中立检索计划。"""

    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class CommandParseStatus(str, Enum):
    """L1 指令解析结果状态；这里只描述解析，不代表执行结果。"""

    MATCHED = "matched"
    INVALID_ARGS = "invalid_args"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class CommandParseResult(BaseModel):
    """系统指令解析产物，由 Gateway S0 透传给后续应用层。"""

    command_id: str | None = None
    raw_input: str
    name: str = ""
    args: FrozenDict[str, Any] = Field(default_factory=FrozenDict)
    tokens: tuple[str, ...] = Field(default_factory=tuple)
    matched_alias: str | None = None
    parse_status: CommandParseStatus
    error: str | None = None

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("args", mode="before")
    @classmethod
    def _freeze_args(cls, value: Any) -> FrozenDict[str, Any]:
        return freeze_mapping(value)

    @property
    def effective_command_id(self) -> str:
        """对外报告的命令标识；解析失败时 ``command_id`` 为空，回退到解析出的命令名。"""
        return self.command_id or self.name or "unknown"


class CommandExecutionStatus(str, Enum):
    """系统指令执行终态。"""

    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    NOT_IMPLEMENTED = "not_implemented"


class CommandExecutionResult(BaseModel):
    """递归不可变的公共系统指令结果。"""

    command_id: str
    status: CommandExecutionStatus
    message: str
    data: FrozenDict[str, Any] = Field(default_factory=FrozenDict)
    client_action: FrozenDict[str, Any] | None = None
    error_code: str | None = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("data", "client_action", mode="before")
    @classmethod
    def _freeze_mapping_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        return freeze_mapping(value)


class GatewayDecision(BaseModel):
    """普通 Gateway flow 面向下游的稳定决策。"""

    target_topic_id: str
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    rewritten_query: str
    search_keywords: tuple[str, ...] = ()
    retrieval_plan: RetrievalPlan
    intent_type: IntentType

    model_config = ConfigDict(frozen=True)


class GatewayCommandOutcome(BaseModel):
    """系统指令终态：Gateway 只解析命令，公开结果只携带解析产物。"""

    kind: Literal["command"] = "command"
    command_parse_result: CommandParseResult

    model_config = ConfigDict(frozen=True)


class GatewayDecisionOutcome(BaseModel):
    """普通决策终态。"""

    kind: Literal["decision"] = "decision"
    decision: GatewayDecision

    model_config = ConfigDict(frozen=True)


type GatewayProcessResult = GatewayCommandOutcome | GatewayDecisionOutcome


class GatewayControlError(RuntimeError):
    """Gateway 请求控制异常基类。"""


class GatewayTimeoutError(GatewayControlError):
    """Gateway 无法在 deadline 内形成完整终态。"""


__all__ = [
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "CommandParseResult",
    "CommandParseStatus",
    "GatewayCommandOutcome",
    "GatewayControlError",
    "GatewayDecision",
    "GatewayDecisionOutcome",
    "GatewayIngressMode",
    "GatewayProcessResult",
    "GatewayTimeoutError",
    "IntentType",
    "RetrievalMode",
    "RetrievalPlan",
]
