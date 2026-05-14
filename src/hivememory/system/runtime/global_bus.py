from hivememory.system.runtime.async_bus import AsyncSystemBus


class GlobalSystemBus(AsyncSystemBus):
    """顶层全局系统总线 (v2, pure async) — 由 HiveMemorySystem 持有，只服务跨子系统公开契约。"""
