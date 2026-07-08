"""Gateway public routes exposed through GlobalSystemBus."""

from hivememory.system.contracts.route_names import RouteNames


class GatewayPublicRoutes:
    """Gateway 对其他子系统暴露的公开路由。"""

    PROCESS = RouteNames.GATEWAY_PROCESS


__all__ = ["GatewayPublicRoutes"]
