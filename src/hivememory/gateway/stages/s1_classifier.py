"""S1 主意图分类 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway import IntentClassifierEngine
from hivememory.engines.gateway.models import IntentType
from hivememory.gateway.pipeline import GatewayState, StageTrace


class IntentClassifierStage:
    """将 IntentClassifierEngine 输出写入 GatewayState。"""

    stage_name = "S1.IntentClassifier"

    def __init__(self, engine: IntentClassifierEngine | None = None) -> None:
        self._engine = engine or IntentClassifierEngine()

    async def process(self, state: GatewayState) -> GatewayState:
        """执行主意图分类，失败时退化为 QUERY。"""

        try:
            result = await self._engine.classify(state.raw_message)
        except Exception as exc:  # pragma: no cover - 防御性 fallback
            state.intent_type = IntentType.QUERY
            state.is_composite = False
            state.stage_trace.append(
                StageTrace(
                    stage_name=self.stage_name,
                    duration_ms=0.0,
                    is_fallback=True,
                    fallback_reason=f"S1 分类失败：{exc}",
                )
            )
            return state

        state.intent_type = IntentType(result.intent_type)
        state.is_composite = result.is_composite
        return state


__all__ = ["IntentClassifierStage"]
