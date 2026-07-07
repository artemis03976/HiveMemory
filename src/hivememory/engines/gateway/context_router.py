"""
Phase 3 Gateway 上下文路由决策原语。

当前实现包裹既有 GatewayEngine，作为 S3 Stage 的初期兼容底座。
"""

from __future__ import annotations

from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.models import ContextRoutingResult


class ContextRouterEngine:
    """Phase 3 S3 上下文路由 engine 骨架。"""

    def __init__(self, gateway_engine: GatewayEngine) -> None:
        self._gateway_engine = gateway_engine

    async def route(
        self,
        message: str,
        *,
        active_topics_menu: str | None = None,
    ) -> ContextRoutingResult:
        """复用现有 GatewayEngine 输出，投影为 ContextRoutingResult。"""

        result = await self._gateway_engine.process(
            message,
            active_topics_menu=active_topics_menu,
        )
        return ContextRoutingResult.from_gateway_result(result)


__all__ = ["ContextRouterEngine"]
