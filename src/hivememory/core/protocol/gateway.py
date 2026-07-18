"""Gateway 跨子系统公共协议。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hivememory.core.models import FrozenDict, freeze_mapping


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


class MemoryWriteSignal(str, Enum):
    """用户输入的记忆价值预判。"""

    WRITE = "WRITE"
    SKIP = "SKIP"
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
    memory_write_signal: MemoryWriteSignal
    retrieval_plan: RetrievalPlan
    intent_type: IntentType

    model_config = ConfigDict(frozen=True)

    @property
    def worth_saving(self) -> bool | None:
        """按公共 memory signal 派生 Patchouli 写入预判。"""

        if self.memory_write_signal == MemoryWriteSignal.WRITE:
            return True
        if self.memory_write_signal == MemoryWriteSignal.SKIP:
            return False
        return None


class GatewayCommandOutcome(BaseModel):
    """系统指令终态。"""

    kind: Literal["command"] = "command"
    command_execution_result: CommandExecutionResult

    model_config = ConfigDict(frozen=True)


class GatewayDecisionOutcome(BaseModel):
    """普通决策终态。"""

    kind: Literal["decision"] = "decision"
    decision: GatewayDecision

    model_config = ConfigDict(frozen=True)


type GatewayProcessResult = GatewayCommandOutcome | GatewayDecisionOutcome


class GatewayControlError(RuntimeError):
    """Gateway 请求控制异常基类。"""


class GatewayCancelledError(GatewayControlError):
    """Gateway 请求被调用方取消。"""


class GatewayTimeoutError(GatewayControlError):
    """Gateway 无法在 deadline 内形成完整终态。"""


__all__ = [
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "GatewayCancelledError",
    "GatewayCommandOutcome",
    "GatewayControlError",
    "GatewayDecision",
    "GatewayDecisionOutcome",
    "GatewayIngressMode",
    "GatewayProcessResult",
    "GatewayTimeoutError",
    "IntentType",
    "MemoryWriteSignal",
    "RetrievalMode",
    "RetrievalPlan",
]
