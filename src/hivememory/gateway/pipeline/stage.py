"""Gateway Stage 协议骨架。"""

from __future__ import annotations

from typing import Any, Protocol


class GatewayStage(Protocol):
    """Phase 3 Pipeline Stage 的最小协议。"""

    async def process(self, state: Any) -> Any:
        ...


__all__ = ["GatewayStage"]
