"""
AliceSystem 集成测试 — 真实 System + GlobalSystemBus 协作

驱动真实 AliceSystem（内部真实装配 AliceRuntime/AliceBridge/AgentRunService）
+ 真实 GlobalSystemBus，零 mock；验证系统 facade start()/stop() 对全局总线的
路由挂载与卸载副作用。
"""

import pytest

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from tests.helpers.operations import OperationsHarness


@pytest.mark.asyncio
async def test_start_registers_public_routes_and_stop_unregisters():
    bus = GlobalSystemBus()
    system = AliceSystem(
        config=HiveMemoryConfig().alice, global_bus=bus, operation_entry=OperationsHarness().entry
    )

    await system.start()

    assert AliceRoutes.RUN_AGENT in bus.list_routes()

    await system.stop()

    assert AliceRoutes.RUN_AGENT not in bus.list_routes()


@pytest.mark.asyncio
async def test_stop_clears_runtime_profile_cache():
    """停止后由 Alice 清理保留的 CALL Profile 派生缓存。"""
    from hivememory.core.contracts.routes import GlobalRoutes
    from hivememory.core.models import AgentProfile, ResolvedAgentProfile
    from tests.helpers.workspace import make_identity_scope

    bus = GlobalSystemBus()

    async def load_profile(_alias, *, identity_scope):
        return ResolvedAgentProfile(profile=AgentProfile(persona="缓存的 Profile"))

    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, load_profile)
    system = AliceSystem(
        config=HiveMemoryConfig().alice, global_bus=bus, operation_entry=OperationsHarness().entry
    )
    await system.start()
    profile = await system.runtime.profile_resolver.resolve(
        "coder", identity_scope=make_identity_scope()
    )
    assert profile.persona == "缓存的 Profile"
    await system.stop()
    assert system.runtime.clear_derived_caches() == 0
