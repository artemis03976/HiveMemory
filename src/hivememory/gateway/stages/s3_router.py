"""S3 上下文路由 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway import ContextRouterEngine
from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentType,
    MemoryWriteSignal,
)
from hivememory.gateway.pipeline import GatewayState, StageTrace
from hivememory.gateway.topic_context import render_topic_snapshots


class ContextRouterStage:
    """调用 ContextRouterEngine，并把路由结果写入 GatewayState。"""

    stage_name = "S3.ContextRouter"

    def __init__(self, engine: ContextRouterEngine) -> None:
        self._engine = engine

    async def process(self, state: GatewayState) -> GatewayState:
        """执行话题路由与查询改写，失败时使用保守 fallback。"""

        try:
            active_topics_menu = None
            if state.session_context.topic_snapshots:
                active_topics_menu = render_topic_snapshots(state.session_context.topic_snapshots)

            result = await self._engine.route(
                state.raw_message,
                active_topics_menu=active_topics_menu,
            )
        except Exception as exc:  # pragma: no cover - 防御性 fallback
            state.topic_id = "NEW_TOPIC"
            state.rewritten_query = state.raw_message
            state.search_keywords = []
            state.memory_write_signal = MemoryWriteSignal.UNKNOWN
            state.stage_trace.append(
                StageTrace(
                    stage_name=self.stage_name,
                    duration_ms=0.0,
                    is_fallback=True,
                    fallback_reason=f"S3 路由失败：{exc}",
                )
            )
            return state

        state.topic_id = result.target_topic
        state.new_topic_title = result.new_topic_title
        state.new_topic_summary = result.new_topic_summary
        state.rewritten_query = result.rewritten_query
        state.search_keywords = list(result.search_keywords)
        state.intent_type = _intent_type_from_gateway_intent(result.intent)
        state.memory_write_signal = (
            MemoryWriteSignal.WRITE if result.worth_saving else MemoryWriteSignal.SKIP
        )
        return state


def _intent_type_from_gateway_intent(intent: GatewayIntent | str) -> IntentType:
    gateway_intent = GatewayIntent(intent)
    if gateway_intent == GatewayIntent.CHAT:
        return IntentType.CHAT
    if gateway_intent == GatewayIntent.SYSTEM:
        return IntentType.UNKNOWN
    return IntentType.QUERY


__all__ = ["ContextRouterStage"]
