"""GatewayRuntime 的本地路由绑定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.components.bus.binding import RouteBinding
from hivememory.gateway.contracts.local_routes import GatewayLocalRoutes

if TYPE_CHECKING:
    from hivememory.gateway.service import GatewayService


def build_gateway_route_bindings(
    service: GatewayService,
) -> tuple[RouteBinding, ...]:
    """从 GatewayService 构造本地路由表。"""

    return (RouteBinding.of(GatewayLocalRoutes.PROCESS, service.process),)


__all__ = ["RouteBinding", "build_gateway_route_bindings"]
