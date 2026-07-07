"""S4b 检索策略预选择 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway import RetrievalStrategyEngine
from hivememory.engines.gateway.models import IntentType, RetrievalMode, RetrievalStrategy
from hivememory.gateway.pipeline import GatewayState, StageTrace


class RetrievalStrategyStage:
    """调用 RetrievalStrategyEngine 产出 retrieval_strategy。"""

    stage_name = "S4b.RetrievalStrategy"

    def __init__(self, engine: RetrievalStrategyEngine | None = None) -> None:
        self._engine = engine or RetrievalStrategyEngine()

    async def process(self, state: GatewayState) -> GatewayState:
        """失败时退化为 HYBRID/top_k=5。"""

        try:
            state.retrieval_strategy = await self._engine.pick(
                intent_type=IntentType(state.intent_type) if state.intent_type else None,
                target_topic=state.topic_id,
            )
        except Exception as exc:  # pragma: no cover - 防御性 fallback
            state.retrieval_strategy = RetrievalStrategy(
                mode=RetrievalMode.HYBRID,
                top_k=5,
                reason="S4b fallback 默认混合检索策略",
            )
            state.stage_trace.append(
                StageTrace(
                    stage_name=self.stage_name,
                    duration_ms=0.0,
                    is_fallback=True,
                    fallback_reason=f"S4b 检索策略失败：{exc}",
                )
            )
        return state


__all__ = ["RetrievalStrategyStage"]
