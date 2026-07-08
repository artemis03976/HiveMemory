"""Gateway 子系统私有总线。"""

from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class GatewayBus(AsyncSystemBus):
    """Gateway 子系统本地总线。"""


__all__ = ["GatewayBus"]
