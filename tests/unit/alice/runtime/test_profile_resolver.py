import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, Identity
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    BusRouteUnavailableError,
    PermissionDeniedError,
)
from tests.helpers.workspace import make_identity_scope


def _make_profile(alias: str = "coder_doll") -> AgentProfile:
    return AgentProfile(persona=f"{alias} persona")


def _identity(user_id: str = "u1", agent_id: str = "omni_doll") -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id)


def _context(user_id: str = "u1", agent_id: str = "omni_doll"):
    return make_identity_scope(actor_identity=_identity(user_id, agent_id))


@pytest.mark.asyncio
async def test_resolve_default_alias_skips_bus():
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = AgentProfileResolver(local_bus=bus)

    profile = await resolver.resolve("omni_doll", identity_scope=_context())

    assert profile is OMNI_DOLL_PROFILE
    bus.request.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_loads_profile_from_bus_and_caches():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=_make_profile("coder_doll"))
    resolver = AgentProfileResolver(local_bus=bus)
    identity_scope = _context()

    first = await resolver.resolve("coder_doll", identity_scope=identity_scope)
    second = await resolver.resolve("coder_doll", identity_scope=identity_scope)

    assert first is second
    assert bus.request.await_count == 1


@pytest.mark.asyncio
async def test_same_actor_profile_cache_is_shared_across_workspaces():
    """捕获 Profile cache 因 Workspace scope 被隐式拆分的缺陷。"""
    class _ProfileBus:
        def __init__(self) -> None:
            self.load_count = 0

        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            self.load_count += 1
            return AgentProfile(persona=f"{alias}:load-{self.load_count}")

    bus = _ProfileBus()
    resolver = AgentProfileResolver(local_bus=bus)
    main = make_identity_scope(
        user_id="u1",
        agent_id="omni_doll",
        workspace_id="main_workspace",
    )
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="omni_doll",
        workspace_id="isolation_workspace",
    )

    first = await resolver.resolve("coder_doll", identity_scope=main)
    second = await resolver.resolve("coder_doll", identity_scope=isolated)

    assert second is first
    assert second.persona == "coder_doll:load-1"
    assert bus.load_count == 1


@pytest.mark.asyncio
async def test_resolve_missing_profile_fails_explicitly():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=None)
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(AliasNotFoundError) as exc_info:
        await resolver.resolve("missing_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.alias.not_found"
    assert exc_info.value.message_key == "mtp.call.profile_not_found"


@pytest.mark.asyncio
async def test_resolve_bus_error_fails_as_service_unavailable():
    bus = MagicMock()
    bus.request = AsyncMock(side_effect=KeyError("route missing"))
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(BusRouteUnavailableError) as exc_info:
        await resolver.resolve("coder_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.system.service_unavailable"


@pytest.mark.asyncio
async def test_resolve_custom_profile_requires_identity_scope():
    """防止 Workspace-sensitive profile 读取在缺 scope 时退回默认身份。"""
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = AgentProfileResolver(local_bus=bus)

    with pytest.raises(ScopeRequiredError):
        await resolver.resolve("coder_doll", identity_scope=None)  # type: ignore[arg-type]

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
        await resolver.resolve("private_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.permission.denied"


@pytest.mark.asyncio
async def test_concurrent_resolves_keep_identity_scoped_cache_entries():
    bus = MagicMock()

    async def load_profile(_route, alias, *, identity_scope):
        await asyncio.sleep(0)
        return AgentProfile(
            persona=f"{alias}:{identity_scope.actor_identity.user_id}"
        )

    bus.request = AsyncMock(side_effect=load_profile)
    resolver = AgentProfileResolver(local_bus=bus)
    first_context = _context("u1")
    second_context = _context("u2")

    first, second = await asyncio.gather(
        resolver.resolve("shared_alias", identity_scope=first_context),
        resolver.resolve("shared_alias", identity_scope=second_context),
    )

    assert first.persona == "shared_alias:u1"
    assert second.persona == "shared_alias:u2"
    assert bus.request.await_count == 2

    cached_first, cached_second = await asyncio.gather(
        resolver.resolve("shared_alias", identity_scope=first_context),
        resolver.resolve("shared_alias", identity_scope=second_context),
    )

    assert cached_first is first
    assert cached_second is second
    assert bus.request.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_same_identity_resolve_loads_once():
    bus = MagicMock()

    async def load_profile(_route, alias, *, identity_scope):
        del identity_scope
        await asyncio.sleep(0)
        return _make_profile(alias)

    bus.request = AsyncMock(side_effect=load_profile)
    resolver = AgentProfileResolver(local_bus=bus)
    identity_scope = _context()

    first, second = await asyncio.gather(
        resolver.resolve("coder_doll", identity_scope=identity_scope),
        resolver.resolve("coder_doll", identity_scope=identity_scope),
    )

    assert first is second
    bus.request.assert_awaited_once()
