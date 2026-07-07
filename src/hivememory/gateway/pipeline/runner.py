"""GatewayPipeline Phase 3A 占位实现。"""

from __future__ import annotations

from collections.abc import Sequence

from hivememory.gateway.pipeline.stage import GatewayStage


class GatewayPipeline:
    """Phase 3B 将在此接入 GatewayState Runner。"""

    def __init__(self, stages: Sequence[GatewayStage] | None = None) -> None:
        self._stages = tuple(stages or ())

    @property
    def stages(self) -> tuple[GatewayStage, ...]:
        """当前已装配的 Stage 列表。"""

        return self._stages


__all__ = ["GatewayPipeline"]
