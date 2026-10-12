"""GatewayPublicRoutes 的类型化路由契约。"""

# route 声明在类体中生成常量，不是实例方法。
# ruff: noqa: N802, N805

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.contracts.route import route

if TYPE_CHECKING:
    from hivememory.core.models import IdentityScope
    from hivememory.core.protocol.gateway import GatewayIngressMode, GatewayProcessResult


class GatewayPublicRoutes:
    """Gateway 对其他子系统暴露的公开路由。"""

    @route("gateway.public.process")
    async def PROCESS(
        message: str,
        *,
        identity_scope: IdentityScope,
        ingress_mode: GatewayIngressMode,
        request_timeout_ms: int | None = None,
    ) -> GatewayProcessResult:
        """把一次 Gateway 请求完整委托给 Runtime 持有的 workflow。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    ALL = (PROCESS,)


__all__ = ["GatewayPublicRoutes"]
