"""ProfileResolver 的单元测试（A2 §2.3 Profile 读取 resolver）。

被测对象：builtin 结果不进缓存、命中按条目随存的源原子 policy 对每个 Actor
逐次授权且不回源、交付副本隔离，以及冷读的代次守护。backing 以内存替身
实现（被测单元边界之外的 Patchouli ``GET_AGENT_PROFILE``）。

resolver 与 backing 位于授权点以下（A1 访问边界返工第 4.1 节）：只流动
授权点组装的可信 ``IdentityScope``，不接收访问 context。
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import pytest

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
        if agent_alias in (None, "default", "omni_doll"):
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
async def test_builtin_profile_is_resolved_by_backing_each_time_and_not_cached():
    """builtin Profile 无源原子：每次经 backing 解析，不进缓存、无本地解析旁路。"""
    backing = _FakeProfileBacking()
    resolver, _, cache = _resolver(backing)

    first = await resolver.get("default", scope=A1)
    second = await resolver.get(None, scope=A1)

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
