"""
Phase 3 Gateway 上下文路由决策原语。

本模块只提供可被 Stage 包裹的纯 engine 骨架，不依赖 GatewayEngine、
SystemBus、Patchouli 或 Alice。真实路由 Prompt 会在后续阶段接入。
"""

from __future__ import annotations

from hivememory.engines.gateway.models import ContextRoutingResult


class ContextRouterEngine:
    """Phase 3 S3 上下文路由 engine 骨架。"""

    async def route(
        self,
        message: str,
        *,
        active_topics_menu: str | None = None,
    ) -> ContextRoutingResult:
        """
        生成最小可用的上下文路由结果。

        Phase 3B 不包裹旧 GatewayEngine；在真实 S3 Prompt 接入前，默认保守
        路由到新话题，并将原始消息作为 rewritten_query。
        """

        _ = active_topics_menu
        return ContextRoutingResult(
            rewritten_query=message,
            search_keywords=[],
            topic_id="NEW_TOPIC",
            new_topic_title=None,
            new_topic_summary=None,
            reason="Phase 3B 默认上下文路由",
        )


__all__ = ["ContextRouterEngine"]
