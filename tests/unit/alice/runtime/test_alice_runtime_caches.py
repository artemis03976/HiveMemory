"""Alice 保留的 CALL Profile 缓存生命周期。"""

import pytest

from hivememory.alice.runtime.core import AliceRuntime
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import AgentProfile, ResolvedAgentProfile
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
async def test_clear_profile_cache_reloads_profile_and_is_idempotent():
    """清理后下一次解析回源，重复清理不会重复统计。"""
    config = HiveMemoryConfig()
    runtime = AliceRuntime(config.alice, config.memory_compiler)
    personas = iter(["旧配置", "新配置"])

    async def profile_route(_alias, *, identity_scope):
        return ResolvedAgentProfile(profile=AgentProfile(persona=next(personas)))

    runtime.local_bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile_route)
    scope = make_identity_scope()
    first = await runtime.profile_resolver.resolve("coder", identity_scope=scope)
    assert first.persona == "旧配置"
    assert runtime.clear_derived_caches() == 1
    assert runtime.clear_derived_caches() == 0
    second = await runtime.profile_resolver.resolve("coder", identity_scope=scope)
    assert second.persona == "新配置"
