"""S5 执行计划占位 Stage。"""

from __future__ import annotations

from hivememory.gateway.pipeline import GatewayState


class PlannerRouterStage:
    """Phase 3C 不生成执行计划，仅保留 Stage 落点。"""

    stage_name = "S5.PlannerRouter"

    async def process(self, state: GatewayState) -> GatewayState:
        """保持 noop。"""

        return state


__all__ = ["PlannerRouterStage"]
