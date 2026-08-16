"""
AliceSystem 集成测试 — 真实 System + GlobalSystemBus 协作

驱动真实 AliceSystem（内部真实装配 AliceRuntime/AliceBridge/AgentRunService）
+ 真实 GlobalSystemBus，零 mock；验证系统 facade start()/stop() 对全局总线的
路由挂载与卸载副作用。
"""

import pytest

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@pytest.mark.asyncio
async def test_start_registers_public_routes_and_stop_unregisters():
    bus = GlobalSystemBus()
    system = AliceSystem(config=HiveMemoryConfig(), global_bus=bus)

    await system.start()

    assert AliceRoutes.RUN_AGENT in bus.list_routes()
    assert AliceRoutes.RUN_AGENT_STREAM in bus.list_routes()

    await system.stop()

    assert AliceRoutes.RUN_AGENT not in bus.list_routes()
    assert AliceRoutes.RUN_AGENT_STREAM not in bus.list_routes()
