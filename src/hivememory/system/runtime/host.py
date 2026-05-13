from hivememory.system.runtime.bus import GlobalSystemBus
from hivememory.system.runtime.registry import SubsystemRegistry


class RuntimeHost:
    """运行时宿主容器 — 持有全局 bus 与子系统注册表。"""

    def __init__(self, bus: GlobalSystemBus, registry: SubsystemRegistry) -> None:
        self.bus = bus
        self.registry = registry
