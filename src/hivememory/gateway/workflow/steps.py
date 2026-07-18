"""Gateway workflow 的原子状态转换结果。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class GatewayStepResult:
    """Step 投影出的单次只读提交。"""

    updates: Mapping[str, Any] = field(default_factory=dict)
    flow_end_reason: str | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "updates", MappingProxyType(dict(self.updates)))


__all__ = ["GatewayStepResult"]
