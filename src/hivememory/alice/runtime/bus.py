from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class AliceBus(AsyncSystemBus):
    """Alice 子系统私有总线。

AliceRuntime 进程内持有，供 Koakuma / AgentProfileResolver 等组件通过
AliceBridge 注册的代理访问 Patchouli 公开能力；不注册全局公开路由。
"""
