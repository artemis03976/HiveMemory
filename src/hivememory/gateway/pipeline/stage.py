"""Gateway Stage 协议骨架。"""

from __future__ import annotations

from typing import Protocol

from hivememory.gateway.pipeline.state import GatewayState, StageResult


class GatewayStage(Protocol):
    """Phase 3 Pipeline Stage 的最小协议。"""

    stage_name: str
    writable_fields: frozenset[str]

    async def process(self, state: GatewayState) -> StageResult:
        ...


__all__ = ["GatewayStage"]
