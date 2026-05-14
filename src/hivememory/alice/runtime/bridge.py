from hivememory.alice.runtime.bus import AliceBus
from hivememory.system.runtime.bridge import SubsystemBridge
from hivememory.system.runtime.global_bus import GlobalSystemBus


class AliceBridge(SubsystemBridge):
    """Alice 子系统桥接器 — 占位，待后续实现。"""

    def __init__(self, local_bus: AliceBus, global_bus: GlobalSystemBus) -> None:
        self._local = local_bus
        self._global = global_bus

    def mount_public_routes(self) -> None:
        pass

    def mount_event_bridges(self) -> None:
        pass

    def unmount(self) -> None:
        pass
