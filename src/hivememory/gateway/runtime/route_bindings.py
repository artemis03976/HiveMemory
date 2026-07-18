"""GatewayRuntime 的本地路由绑定。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hivememory.gateway.contracts.local_routes import GatewayLocalRoutes

if TYPE_CHECKING:
    from hivememory.gateway.service import GatewayService

RouteBinding = tuple[str, Callable[..., Any]]


def build_gateway_route_bindings(
    service: GatewayService,
) -> tuple[RouteBinding, ...]:
    """从 GatewayService 构造本地路由表。"""

    return ((GatewayLocalRoutes.PROCESS, service.process),)


__all__ = ["RouteBinding", "build_gateway_route_bindings"]
