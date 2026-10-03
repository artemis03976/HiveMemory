"""GatewayService：Gateway 子系统业务入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayProcessResult,
)
from hivememory.gateway.runtime import GatewayRuntime

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext


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
        identity_scope: IdentityScope,
        ingress_mode: GatewayIngressMode,
        request_timeout_ms: int | None = None,
        access: WorkspaceAccessContext | None = None,
    ) -> GatewayProcessResult:
        """把一次 Gateway 请求完整委托给 Runtime 持有的 workflow。

        ``access`` 是调用方（主动链路为任务进程）绑定的访问 context：Gateway
        不做授权判断，只把它原样传给 Patchouli 的话题读取路由（A1 访问边界
        返工第 4.3 节）；被动链路等无 Actor context 的调用方传 ``None``，
        话题读取失败按既有保守降级处理。
        """

        configured_timeout_ms = self._runtime.config.workflow.default_request_timeout_ms
        effective_timeout_ms = (
            configured_timeout_ms
            if request_timeout_ms is None
            else min(request_timeout_ms, configured_timeout_ms)
        )

        identity_scope = require_identity_scope(identity_scope)
        return await self._runtime.workflow.run(
            message,
            identity_scope=identity_scope,
            ingress_mode=ingress_mode,
            request_timeout_ms=effective_timeout_ms,
            access=access,
        )


__all__ = ["GatewayService"]
