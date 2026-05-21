from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.core.models import AgentProfile, OMNI_DOLL_PROFILE
from hivememory.system.contracts.routes import GlobalRoutes


def _make_profile(alias: str = "coder_doll") -> AgentProfile:
    return AgentProfile(persona=f"{alias} persona")


@pytest.mark.asyncio
async def test_resolve_default_alias_skips_bus():
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = AgentProfileResolver(local_bus=bus)

    profile = await resolver.resolve("omni_doll")

    assert profile is OMNI_DOLL_PROFILE
    bus.request.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_loads_profile_from_bus_and_caches():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=_make_profile("coder_doll"))
    resolver = AgentProfileResolver(local_bus=bus)

    first = await resolver.resolve("coder_doll")
    second = await resolver.resolve("coder_doll")

    assert first is second
    assert first.persona == "coder_doll persona"
    bus.request.assert_awaited_once_with(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        "coder_doll",
    )


@pytest.mark.asyncio
async def test_resolve_missing_profile_falls_back_to_default():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=None)
    resolver = AgentProfileResolver(local_bus=bus)

    profile = await resolver.resolve("missing_doll")

    assert profile is OMNI_DOLL_PROFILE


@pytest.mark.asyncio
async def test_resolve_bus_error_falls_back_to_default():
    bus = MagicMock()
    bus.request = AsyncMock(side_effect=KeyError("route missing"))
    resolver = AgentProfileResolver(local_bus=bus)

    profile = await resolver.resolve("coder_doll")

    assert profile is OMNI_DOLL_PROFILE

