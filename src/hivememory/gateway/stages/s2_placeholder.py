"""S2 复合意图占位 Stage。"""

from __future__ import annotations

from hivememory.gateway.pipeline import GatewayState


class CompositePlaceholderStage:
    """Phase 3 只记录复合意图延后处理，不做分解。"""

    stage_name = "S2.CompositePlaceholder"

    async def process(self, state: GatewayState) -> GatewayState:
        """记录 Phase 4 前的复合意图占位信号。"""

        if state.is_composite:
            state.composite_deferred = True
            state.composite_deferred_reason = "Phase 3C 暂不执行复合意图分解"
        return state


__all__ = ["CompositePlaceholderStage"]
