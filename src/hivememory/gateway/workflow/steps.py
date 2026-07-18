"""Gateway workflow 的原子状态转换结果。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hivememory.gateway.workflow.state import GatewayStateSnapshot

class RecoverableGatewayError(Exception):
    """Provider、Engine 或 Resolver adapter 的预期能力失败。"""


@dataclass(frozen=True)
class GatewayStepResult:
    """Step 投影出的单次只读提交。"""

    updates: Mapping[str, Any] = field(default_factory=dict)
    flow_end_reason: str | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "updates", MappingProxyType(dict(self.updates)))


@dataclass(frozen=True)
class GatewayWorkflowStep[InputT, OutputT]:
    """固定拓扑中唯一的通用原子调度单元。"""

    step_id: str
    select_input: Callable[[GatewayStateSnapshot], InputT]
    invoke: Callable[[InputT], Awaitable[OutputT]]
    project: Callable[[OutputT], Mapping[str, Any]]
    timeout_ms: int | None = None
    fallback: Callable[[InputT, Exception], Mapping[str, Any]] | None = None
    resolve_flow_end: Callable[[OutputT], str | None] | None = None

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("Gateway workflow step_id 不能为空")
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("Gateway workflow step timeout_ms 必须大于 0")


__all__ = [
    "GatewayStepResult",
    "GatewayWorkflowStep",
    "RecoverableGatewayError",
]
