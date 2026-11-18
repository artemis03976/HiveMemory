"""ProfileResolver（Profile 读取器）的单元测试。

被测对象：每次解析与列出都经 backing 读取当前的 Profile，不缓存，Profile 文件的改动在
下一次读取时生效；读取视图关闭后显式失败。backing 以内存替身实现（被测单元边界之外的
Patchouli ``GET_AGENT_PROFILE`` 与 ``AGENT_PROFILE_LIST``）。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hivememory.core.errors import ResourceNotFoundError, ResourceUnavailableError
from hivememory.core.models import AgentProfile, StoredAgentProfile
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.resolution.guard import ColdReadGuard
from hivememory.workspace.resolution.profile import ProfileResolver


class _FakeProfileBacking:
    """内存版 Profile backing：按 agent_id 返回当前预置的 Profile。"""

    def __init__(self) -> None:
        self.profiles: dict[str, AgentProfile] = {}

    async def get_agent_profile(self, agent_alias):
        profile = self.profiles.get(agent_alias)
        if profile is None:
            raise ResourceNotFoundError()
        return profile

    async def list_agent_profiles(self):
        saved_at = datetime(2026, 11, 18, tzinfo=UTC)
        return [
            StoredAgentProfile(profile=profile, saved_at=saved_at)
            for profile in self.profiles.values()
        ]


def _resolver(backing) -> tuple[ProfileResolver, ColdReadGuard]:
    guard = ColdReadGuard(WorkspaceEpochs())
    return ProfileResolver(guard=guard, backing=backing), guard


@pytest.mark.asyncio
async def test_changed_profiles_are_read_on_the_next_resolution_and_listing():
    """不缓存：Profile 被改动、新增或删除后，下一次解析与列出就反映出来。"""
    backing = _FakeProfileBacking()
    backing.profiles["coder_doll"] = AgentProfile(agent_id="coder_doll", persona="原来的")
    resolver, _ = _resolver(backing)
    assert (await resolver.get("coder_doll")).persona == "原来的"

    backing.profiles["coder_doll"] = AgentProfile(agent_id="coder_doll", persona="改过的")
    backing.profiles["reviewer"] = AgentProfile(agent_id="reviewer")

    assert (await resolver.get("coder_doll")).persona == "改过的"
    assert [profile.agent_id for profile in await resolver.list()] == [
        "coder_doll",
        "reviewer",
    ]

    del backing.profiles["coder_doll"]
    with pytest.raises(ResourceNotFoundError):
        await resolver.get("coder_doll")


@pytest.mark.asyncio
async def test_closed_read_view_rejects_profile_reads():
    backing = _FakeProfileBacking()
    backing.profiles["coder_doll"] = AgentProfile(agent_id="coder_doll")
    resolver, guard = _resolver(backing)
    guard.close()

    with pytest.raises(ResourceUnavailableError):
        await resolver.get("coder_doll")
    with pytest.raises(ResourceUnavailableError):
        await resolver.list()
