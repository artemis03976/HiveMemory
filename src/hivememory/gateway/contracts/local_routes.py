"""Gateway 子系统内部路由常量。"""


class GatewayLocalRoutes:
    """仅供 GatewayBus 本地挂载与调用的路由。"""

    PROCESS = "gateway.process"

    ALL = (PROCESS,)


__all__ = ["GatewayLocalRoutes"]
