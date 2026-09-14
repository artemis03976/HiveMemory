"""通过 GlobalSystemBus 暴露的 Gateway 公开路由。"""

from hivememory.system.contracts.route_names import RouteNames


class GatewayPublicRoutes:
    """Gateway 对其他子系统暴露的公开路由。"""

    PROCESS = RouteNames.GATEWAY_PROCESS


__all__ = ["GatewayPublicRoutes"]
