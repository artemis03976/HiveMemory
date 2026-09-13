import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.alice.runtime.profile_resolver import (
    AgentProfileCache,
    AgentProfileResolver,
)
from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, AgentProfile
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    BusRouteUnavailableError,
    PermissionDeniedError,
)
from tests.helpers.workspace import make_identity_scope


def _make_profile(alias: str = "coder_doll") -> AgentProfile:
    return AgentProfile(persona=f"{alias} persona")


def _identity(user_id: str = "u1", agent_id: str = "omni_doll") -> ActorIdentity:
    return ActorIdentity(user_id=user_id, agent_id=agent_id)


def _context(user_id: str = "u1", agent_id: str = "omni_doll"):
    return make_identity_scope(actor_identity=_identity(user_id, agent_id))


def _resolver(bus) -> AgentProfileResolver:
    return AgentProfileResolver(local_bus=bus, profile_cache=AgentProfileCache())


@pytest.mark.asyncio
async def test_resolve_default_alias_skips_bus():
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = _resolver(bus)

    profile = await resolver.resolve("omni_doll", identity_scope=_context())

    assert profile is OMNI_DOLL_PROFILE
    bus.request.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_loads_profile_from_bus_and_caches():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=_make_profile("coder_doll"))
    resolver = _resolver(bus)
    identity_scope = _context()

    first = await resolver.resolve("coder_doll", identity_scope=identity_scope)
    second = await resolver.resolve("coder_doll", identity_scope=identity_scope)

    assert first is second
    assert bus.request.await_count == 1


@pytest.mark.asyncio
async def test_same_actor_same_alias_caches_per_workspace():
    """同 Actor 同 alias 在不同 Workspace 各自缓存各自的 profile，互不串扰。"""
    class _ProfileBus:
        def __init__(self) -> None:
            self.load_count = 0

        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            self.load_count += 1
            return AgentProfile(persona=f"{alias}:load-{self.load_count}")

    bus = _ProfileBus()
    resolver = _resolver(bus)
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

    # 同名 profile 不跨 Workspace 复用：各自独立加载。
    assert second is not first
    assert first.persona == "coder_doll:load-1"
    assert second.persona == "coder_doll:load-2"
    assert bus.load_count == 2

    # 各自重复解析命中各自 Workspace 的条目，不再触发加载。
    assert await resolver.resolve("coder_doll", identity_scope=main) is first
    assert await resolver.resolve("coder_doll", identity_scope=isolated) is second
    assert bus.load_count == 2


@pytest.mark.asyncio
async def test_same_workspace_different_team_caches_separately():
    """同 Workspace 同 alias，team 不同的执行者各自缓存，不互相复用。"""
    class _ProfileBus:
        def __init__(self) -> None:
            self.load_count = 0

        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            self.load_count += 1
            return AgentProfile(persona=f"{alias}:load-{self.load_count}")

    bus = _ProfileBus()
    resolver = _resolver(bus)
    default_team = make_identity_scope(
        actor_identity=ActorIdentity(user_id="u1", agent_id="omni_doll"),
        workspace_id="main_workspace",
    )
    team_a = make_identity_scope(
        actor_identity=ActorIdentity(
            user_id="u1",
            agent_id="omni_doll",
            team_id="team-a",
        ),
        workspace_id="main_workspace",
    )

    first = await resolver.resolve("coder_doll", identity_scope=default_team)
    second = await resolver.resolve("coder_doll", identity_scope=team_a)

    assert second is not first
    assert bus.load_count == 2
    assert await resolver.resolve("coder_doll", identity_scope=default_team) is first
    assert await resolver.resolve("coder_doll", identity_scope=team_a) is second
    assert bus.load_count == 2


@pytest.mark.asyncio
async def test_session_id_does_not_fragment_cache():
    """session_id 是兼容字段，不参与 cache key，不造成按会话碎片化。"""
    class _ProfileBus:
        def __init__(self) -> None:
            self.load_count = 0

        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            self.load_count += 1
            return AgentProfile(persona=f"{alias}:load-{self.load_count}")

    bus = _ProfileBus()
    resolver = _resolver(bus)
    with_session = make_identity_scope(
        actor_identity=ActorIdentity(
            user_id="u1",
            agent_id="omni_doll",
            session_id="sess-1",
        ),
        workspace_id="main_workspace",
    )
    other_session = make_identity_scope(
        actor_identity=ActorIdentity(
            user_id="u1",
            agent_id="omni_doll",
            session_id="sess-2",
        ),
        workspace_id="main_workspace",
    )

    first = await resolver.resolve("coder_doll", identity_scope=with_session)
    second = await resolver.resolve("coder_doll", identity_scope=other_session)

    assert second is first
    assert bus.load_count == 1


@pytest.mark.asyncio
async def test_missing_profile_failure_is_not_cached():
    """Profile 缺失错误不进入缓存，同一坐标随后可重新加载并缓存。"""
    loads: list[AgentProfile | None] = [None]

    class _ProfileBus:
        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            result = loads.pop(0) if loads else _make_profile(alias)
            return result

    bus = _ProfileBus()
    resolver = _resolver(bus)
    identity_scope = _context()

    with pytest.raises(AliasNotFoundError):
        await resolver.resolve("coder_doll", identity_scope=identity_scope)

    recovered = await resolver.resolve("coder_doll", identity_scope=identity_scope)

    assert recovered == _make_profile("coder_doll")
    # 成功结果进入缓存，第三次解析不再触发加载。
    assert await resolver.resolve("coder_doll", identity_scope=identity_scope) is recovered


@pytest.mark.asyncio
async def test_permission_denied_profile_load_is_not_cached():
    """越权错误不进入缓存，授权恢复后同一坐标可正常加载并缓存。"""
    denials: list[bool] = [True]

    class _ProfileBus:
        def __init__(self) -> None:
            self.load_count = 0

        async def request(self, _route, alias, *, identity_scope):
            del identity_scope
            self.load_count += 1
            if denials and denials.pop(0):
                raise PermissionDeniedError(
                    message_key="mtp.call.profile_permission_denied",
                    params={"agent_alias": alias},
                )
            return _make_profile(alias)

    bus = _ProfileBus()
    resolver = _resolver(bus)
    identity_scope = _context()

    with pytest.raises(PermissionDeniedError):
        await resolver.resolve("private_doll", identity_scope=identity_scope)

    recovered = await resolver.resolve("private_doll", identity_scope=identity_scope)

    assert recovered == _make_profile("private_doll")
    assert bus.load_count == 2
    assert await resolver.resolve("private_doll", identity_scope=identity_scope) is recovered
    assert bus.load_count == 2


@pytest.mark.asyncio
async def test_resolve_missing_profile_fails_explicitly():
    bus = MagicMock()
    bus.request = AsyncMock(return_value=None)
    resolver = _resolver(bus)

    with pytest.raises(AliasNotFoundError) as exc_info:
        await resolver.resolve("missing_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.alias.not_found"
    assert exc_info.value.message_key == "mtp.call.profile_not_found"


@pytest.mark.asyncio
async def test_resolve_bus_error_fails_as_service_unavailable():
    bus = MagicMock()
    bus.request = AsyncMock(side_effect=KeyError("route missing"))
    resolver = _resolver(bus)

    with pytest.raises(BusRouteUnavailableError) as exc_info:
        await resolver.resolve("coder_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.system.service_unavailable"


@pytest.mark.asyncio
async def test_resolve_custom_profile_requires_identity_scope():
    """防止 Workspace-sensitive profile 读取在缺 scope 时退回默认身份。"""
    bus = MagicMock()
    bus.request = AsyncMock()
    resolver = _resolver(bus)

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
    resolver = _resolver(bus)

    with pytest.raises(PermissionDeniedError) as exc_info:
        await resolver.resolve("private_doll", identity_scope=_context())

    assert exc_info.value.code == "mtp.permission.denied"


@pytest.mark.asyncio
async def test_concurrent_resolves_keep_identity_scoped_cache_entries():
    """同 Workspace 不同 Actor 并发解析各自加载并命中各自条目，不互相污染。"""
    bus = MagicMock()

    async def load_profile(_route, alias, *, identity_scope):
        await asyncio.sleep(0)
        return AgentProfile(
            persona=f"{alias}:{identity_scope.actor_identity.user_id}"
        )

    bus.request = AsyncMock(side_effect=load_profile)
    resolver = _resolver(bus)
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
    resolver = _resolver(bus)
    identity_scope = _context()

    first, second = await asyncio.gather(
        resolver.resolve("coder_doll", identity_scope=identity_scope),
        resolver.resolve("coder_doll", identity_scope=identity_scope),
    )

    assert first is second
    bus.request.assert_awaited_once()
