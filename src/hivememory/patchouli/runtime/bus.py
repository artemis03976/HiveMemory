from hivememory.system.runtime.async_bus import AsyncSystemBus


class PatchouliBus(AsyncSystemBus):
    """Patchouli 子系统私有总线 — 仅服务记忆域内部对象与运行时协作。"""
