from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.registry import SubsystemRegistry
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


class RuntimeHost:
    """运行时宿主容器 — 持有全局 bus、子系统注册表与全局维护调度器。"""

    def __init__(
        self,
        bus: GlobalSystemBus,
        registry: SubsystemRegistry,
        scheduler: GlobalMaintenanceScheduler,
    ) -> None:
        self.bus = bus
        self.registry = registry
        self.scheduler = scheduler
