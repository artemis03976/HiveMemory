from abc import ABC, abstractmethod


class SubsystemBridge(ABC):
    """子系统总线桥接抽象 — 连接子系统私有总线与全局总线。"""

    @abstractmethod
    def mount_public_routes(self) -> None: ...

    @abstractmethod
    def mount_event_bridges(self) -> None: ...

    @abstractmethod
    def unmount(self) -> None: ...

    def mount(self) -> None:
        self.mount_public_routes()
        self.mount_event_bridges()
