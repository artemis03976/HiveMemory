"""ProfileResolver 的单元测试（A2 §2.3 Profile 读取 resolver）。

被测对象：builtin 结果不进缓存、命中按条目随存的源原子 policy 对每个 Actor
逐次授权且不回源、交付副本隔离，以及冷读的代次守护。backing 以内存替身
实现（被测单元边界之外的 Patchouli ``GET_AGENT_PROFILE``）。

resolver 与 backing 位于授权点以下（A1 访问边界返工第 4.1 节）：只流动
授权点组装的可信 ``IdentityScope``，不接收访问 context。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    MemoryAccessPolicy,
    MemoryVisibility,
    ResolvedAgentProfile,
)
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.cache.profile import ProfileCache
from hivememory.workspace.resolution.guard import ColdReadGuard
from hivememory.workspace.resolution.profile import ProfileResolver
from tests.helpers.workspace import make_identity_scope

A1 = make_identity_scope(user_id="u1", agent_id="a1")
A2 = make_identity_scope(user_id="u1", agent_id="a2")
MAIN = A1.workspace_identity


def _resolved(agent_id: str, *, persona: str = "persona", private_to: str | None = None):
    policy = (
        MemoryAccessPolicy(visibility=MemoryVisibility.PRIVATE, target_agent_id=private_to)
        if private_to
        else MemoryAccessPolicy.public()
    )
    return ResolvedAgentProfile(
        profile=AgentProfile(agent_id=agent_id, persona=persona),
        access_policy=policy,
        source_memory_id=uuid4(),
        source_version=1,
    )


class _FakeProfileBacking:
    """内存版 Profile backing：按 alias 返回预置解析结果，builtin 规则同 Patchouli。"""

    def __init__(self) -> None:
        self.profiles: dict[str, ResolvedAgentProfile] = {}
        self.calls = 0
        self.on_fetch: Callable[[], None] | None = None

    async def get_agent_profile(self, agent_alias, *, scope):
        self.calls += 1
        if self.on_fetch is not None:
            self.on_fetch()
        if agent_alias in (None, "", "default", "omni_doll"):
            return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE.model_copy(deep=True))
        resolved = self.profiles.get(agent_alias)
        if resolved is None:
            raise AliasNotFoundError(message_key="mtp.call.profile_not_found")
        return resolved.model_copy(deep=True)


def _resolver(backing) -> tuple[ProfileResolver, WorkspaceEpochs, ProfileCache]:
    epochs = WorkspaceEpochs()
    cache = ProfileCache(capacity=16)
    resolver = ProfileResolver(cache=cache, guard=ColdReadGuard(epochs), backing=backing)
    return resolver, epochs, cache


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", [None, "", "default", "omni_doll"])
async def test_builtin_profile_is_resolved_by_backing_each_time_and_not_cached(alias):
    """builtin Profile 无源原子：每次经 backing 解析，不进缓存、无本地解析旁路。"""
    backing = _FakeProfileBacking()
    resolver, _, cache = _resolver(backing)

    first = await resolver.get(alias, scope=A1)
    second = await resolver.get(alias, scope=A1)

    assert first == OMNI_DOLL_PROFILE
    assert second == OMNI_DOLL_PROFILE
    assert (backing.calls, cache.size) == (2, 0)


@pytest.mark.asyncio
async def test_cached_profile_is_authorized_per_actor_by_source_policy():
    """命中按源原子 policy 对每个 Actor 授权：不可见者按"不存在"失败，且不回源。

    捕获缓存"已通过授权的裸 AgentProfile"、让其他 Actor 借共享条目读取私有
    Profile 的缺陷。
    """
    backing = _FakeProfileBacking()
    backing.profiles["private_doll"] = _resolved("private_doll", private_to="a1")
    resolver, _, _ = _resolver(backing)
    await resolver.get("private_doll", scope=A1)

    with pytest.raises(AliasNotFoundError):
        await resolver.get("private_doll", scope=A2)
    owner = await resolver.get("private_doll", scope=A1)

    assert (owner.agent_id, backing.calls) == ("private_doll", 1)


@pytest.mark.asyncio
async def test_hit_delivers_isolated_profile_copy():
    """调用方修改交付的 Profile 不影响后续读取。"""
    backing = _FakeProfileBacking()
    backing.profiles["coder_doll"] = _resolved("coder_doll", persona="original")
    resolver, _, _ = _resolver(backing)

    first = await resolver.get("coder_doll", scope=A1)
    first.persona = "mutated"
    again = await resolver.get("coder_doll", scope=A1)

    assert (again.persona, backing.calls) == ("original", 1)


@pytest.mark.asyncio
async def test_profile_read_during_workspace_change_is_retried_before_caching():
    """冷读期间代次变化：旧解析结果不回填，重读后的当前结果进入缓存。"""
    backing = _FakeProfileBacking()
    backing.profiles["coder_doll"] = _resolved("coder_doll", persona="v1")
    resolver, epochs, _ = _resolver(backing)

    def concurrent_update() -> None:
        backing.on_fetch = None
        backing.profiles["coder_doll"] = _resolved("coder_doll", persona="v2")
        epochs.advance(MAIN)

    backing.on_fetch = concurrent_update
    result = await resolver.get("coder_doll", scope=A1)
    cached = await resolver.get("coder_doll", scope=A1)

    assert (result.persona, cached.persona, backing.calls) == ("v2", "v2", 2)


@pytest.mark.asyncio
async def test_same_profile_alias_is_cached_separately_for_each_workspace():
    """同一 Actor 的同名 Profile 在不同 Workspace 各自解析，不能串用缓存。"""
    isolated = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="isolated")

    class WorkspaceBacking:
        async def get_agent_profile(self, alias, *, scope):
            return _resolved(alias, persona=scope.workspace_identity.workspace_id)

    resolver, _, _ = _resolver(WorkspaceBacking())
    main_profile = await resolver.get("coder_doll", scope=A1)
    isolated_profile = await resolver.get("coder_doll", scope=isolated)
    main_again = await resolver.get("coder_doll", scope=A1)

    assert (main_profile.persona, isolated_profile.persona, main_again.persona) == (
        "main_workspace",
        "isolated",
        "main_workspace",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ResourceUnavailableError("存储不可达"), asyncio.CancelledError()]
)
async def test_failed_profile_read_does_not_cache_and_next_read_can_recover(failure):
    """不可达与取消均不写缓存；恢复后重新读取并缓存当前 Profile。"""

    class RecoveringBacking(_FakeProfileBacking):
        first_read = True

        async def get_agent_profile(self, alias, *, scope):
            if self.first_read:
                self.first_read = False
                raise failure
            return await super().get_agent_profile(alias, scope=scope)

    backing = RecoveringBacking()
    backing.profiles["coder_doll"] = _resolved("coder_doll", persona="recovered")
    resolver, _, _ = _resolver(backing)
    with pytest.raises(type(failure)):
        await resolver.get("coder_doll", scope=A1)
    recovered = await resolver.get("coder_doll", scope=A1)
    backing.profiles["coder_doll"] = _resolved("coder_doll", persona="uncached replacement")
    cached = await resolver.get("coder_doll", scope=A1)

    assert (recovered.persona, cached.persona, backing.calls) == ("recovered", "recovered", 1)


@pytest.mark.asyncio
async def test_missing_profile_is_reloaded_after_it_becomes_available():
    """缺失结果不驻留缓存，创建同名图纸后可以重新解析。"""
    backing = _FakeProfileBacking()
    resolver, _, _ = _resolver(backing)
    with pytest.raises(AliasNotFoundError):
        await resolver.get("coder_doll", scope=A1)
    backing.profiles["coder_doll"] = _resolved("coder_doll", persona="newly created")

    recovered = await resolver.get("coder_doll", scope=A1)

    assert (recovered.persona, backing.calls) == ("newly created", 2)
