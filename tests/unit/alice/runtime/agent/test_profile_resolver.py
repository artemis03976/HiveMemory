import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, Identity
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    BusRouteUnavailableError,
    PermissionDeniedError,
)
from hivememory.system.contracts.routes import GlobalRoutes


def _make_profile(alias: str = "coder_doll") -> AgentProfile:
    return AgentProfile(persona=f"{alias} persona")


def _identity(user_id: str = "u1", agent_id: str = "omni_doll") -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id)


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
    identity = _identity()

    first = await resolver.resolve("coder_doll", identity=identity)
    second = await resolver.resolve("coder_doll", identity=identity)

    assert first is second
    assert first.persona == "coder_doll persona"
    bus.request.assert_awaited_once_with(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        "coder_doll",
        identity=identity,
    )


@pytest.mark.asyncio
async def test_resolve_missing_profile_fails_explicitly():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=None)
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(AliasNotFoundError) as exc_info:
        await resolver.resolve("missing_doll", identity=_identity())

    assert exc_info.value.code == "mtp.alias.not_found"
    assert exc_info.value.message_key == "mtp.call.profile_not_found"


@pytest.mark.asyncio
async def test_resolve_bus_error_fails_as_service_unavailable():
    bus = MagicMock()
    bus.request = AsyncMock(side_effect=KeyError("route missing"))
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(BusRouteUnavailableError) as exc_info:
        await resolver.resolve("coder_doll", identity=_identity())

    assert exc_info.value.code == "mtp.system.service_unavailable"


@pytest.mark.asyncio
async def test_resolve_custom_profile_requires_identity():
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(PermissionDeniedError) as exc_info:
        await resolver.resolve("coder_doll")

    assert exc_info.value.message_key == "mtp.call.profile_permission_denied"
    bus.request.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_propagates_profile_permission_denial():
    bus = MagicMock()
    denial = PermissionDeniedError(
        message_key="mtp.call.profile_permission_denied",
        params={"agent_alias": "private_doll"},
    )
    bus.request = AsyncMock(side_effect=denial)
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(PermissionDeniedError) as exc_info:
        await resolver.resolve("private_doll", identity=_identity())

    assert exc_info.value is denial


@pytest.mark.asyncio
async def test_concurrent_resolves_keep_identity_scoped_cache_entries():
    bus = MagicMock()

    async def load_profile(_route, alias, *, identity):
        await asyncio.sleep(0)
        return AgentProfile(persona=f"{alias}:{identity.user_id}")

    bus.request = AsyncMock(side_effect=load_profile)
    resolver = AgentProfileResolver(local_bus=bus)
    first_identity = _identity("u1")
    second_identity = _identity("u2")

    first, second = await asyncio.gather(
        resolver.resolve("shared_alias", identity=first_identity),
        resolver.resolve("shared_alias", identity=second_identity),
    )

    assert first.persona == "shared_alias:u1"
    assert second.persona == "shared_alias:u2"
    assert bus.request.await_count == 2

    cached_first, cached_second = await asyncio.gather(
        resolver.resolve("shared_alias", identity=first_identity),
        resolver.resolve("shared_alias", identity=second_identity),
    )

    assert cached_first is first
    assert cached_second is second
    assert bus.request.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_same_identity_resolve_loads_once():
    bus = MagicMock()

    async def load_profile(_route, alias, *, identity):
        await asyncio.sleep(0)
        return _make_profile(alias)

    bus.request = AsyncMock(side_effect=load_profile)
    resolver = AgentProfileResolver(local_bus=bus)
    identity = _identity()

    first, second = await asyncio.gather(
        resolver.resolve("coder_doll", identity=identity),
        resolver.resolve("coder_doll", identity=identity),
    )

    assert first is second
    bus.request.assert_awaited_once()

