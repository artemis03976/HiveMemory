"""GatewayService：Gateway 子系统业务入口。"""

from __future__ import annotations

import asyncio

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayProcessResult,
)
from hivememory.gateway.runtime import GatewayRuntime


class GatewayService:
    """
    Gateway 子系统业务门面。

    调用方只通过 process 进入 Gateway Workflow。
    """

    def __init__(self, runtime: GatewayRuntime) -> None:
        self._runtime = runtime

    async def process(
        self,
        message: str,
        *,
        identity: Identity,
        ingress_mode: GatewayIngressMode,
        cancel_event: asyncio.Event | None = None,
        request_timeout_ms: int | None = None,
    ) -> GatewayProcessResult:
        """把一次 Gateway 请求完整委托给 Runtime 持有的 workflow。"""

        configured_timeout_ms = self._runtime.config.workflow.default_request_timeout_ms
        effective_timeout_ms = (
            configured_timeout_ms
            if request_timeout_ms is None
            else min(request_timeout_ms, configured_timeout_ms)
        )

        return await self._runtime.workflow.run(
            message,
            identity=identity,
            ingress_mode=ingress_mode,
            cancel_event=cancel_event,
            request_timeout_ms=effective_timeout_ms,
        )


__all__ = ["GatewayService"]
