"""Gateway Stage 协议骨架。"""

from __future__ import annotations

from typing import Protocol

from hivememory.gateway.pipeline.state import GatewayState


class GatewayStage(Protocol):
    """Phase 3 Pipeline Stage 的最小协议。"""

    async def process(self, state: GatewayState) -> GatewayState:
        ...


__all__ = ["GatewayStage"]
