"""GatewayPipeline 最小 Runner。"""

from __future__ import annotations

from collections.abc import Sequence
from time import monotonic

from hivememory.gateway.context import SessionContext
from hivememory.gateway.pipeline.stage import GatewayStage
from hivememory.gateway.pipeline.state import GatewayState


class GatewayPipeline:
    """按顺序执行 Gateway Stage，并在结束时封印 GatewayState。"""

    def __init__(self, stages: Sequence[GatewayStage] | None = None) -> None:
        self._stages = tuple(stages or ())

    @property
    def stages(self) -> tuple[GatewayStage, ...]:
        """当前已装配的 Stage 列表。"""

        return self._stages

    async def run(self, message: str, context: SessionContext) -> GatewayState:
        """执行 Pipeline，返回封印后的 GatewayState。"""

        state = GatewayState(raw_message=message, session_context=context)
        return await self.run_state(state)

    async def run_state(self, state: GatewayState) -> GatewayState:
        """从已有 GatewayState 继续执行 Pipeline。"""

        for stage in self._stages:
            stage_name = self._stage_name(stage)
            start = monotonic()
            result = await stage.process(state)
            duration_ms = (monotonic() - start) * 1000
            state.apply_stage_result(
                stage_name=stage_name,
                result=result,
                duration_ms=duration_ms,
                writable_fields=self._writable_fields(stage),
            )
            if result.flow_ended:
                return state.seal()

        return state.seal()

    @staticmethod
    def _stage_name(stage: GatewayStage) -> str:
        return getattr(stage, "stage_name", stage.__class__.__name__)

    @staticmethod
    def _writable_fields(stage: GatewayStage) -> frozenset[str]:
        return getattr(stage, "writable_fields", frozenset())


__all__ = ["GatewayPipeline"]
