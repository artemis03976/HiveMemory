from hivememory.infrastructure.system_bus import SystemBus


class GlobalSystemBus(SystemBus):
    """顶层系统总线 — 顶层 system 显式持有的全局通信骨架。"""


__all__ = ["GlobalSystemBus", "SystemBus"]
