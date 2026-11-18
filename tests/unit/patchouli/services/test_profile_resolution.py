"""RetrievalFamiliar.get_agent_profile 的单元测试（Profile 解析 backing）。

被测对象：Profile 解析的唯一实现，读取真实的 AgentProfileStore（临时目录）：
- 未指定或选择内置 alias 时返回内置 omni_doll，每次都是独立副本；
- 其余 agent_id 读取 Profile 文件，以文件名为 ``agent_id``；
- 缺失、文件不合格与已停用分别显式失败，不降级为内置 Profile。
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile
from hivememory.memory_library.agent_profiles import AgentProfileStore
from hivememory.patchouli.contracts.errors import ProfileResolutionError
from hivememory.patchouli.services.retrieval import RetrievalFamiliar


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    return AgentProfileStore(tmp_path)


@pytest.fixture
def familiar(store):
    library = Mock()
    library.agent_profiles = store
    return RetrievalFamiliar(engine=Mock(), memory_library=library)


@pytest.mark.parametrize("alias", [None, "   ", "default", "omni_doll"])
def test_builtin_alias_returns_a_fresh_builtin_profile(familiar, alias):
    resolved = _run(familiar.get_agent_profile(alias))

    assert resolved == OMNI_DOLL_PROFILE
    resolved.persona = "调用方的修改"
    assert OMNI_DOLL_PROFILE.persona == ""


def test_profile_file_is_resolved_by_agent_id(familiar, store):
    profile = AgentProfile(agent_id="coder_doll", title="写代码的", persona="人设", model="m")
    store.save(profile)

    assert _run(familiar.get_agent_profile(" coder_doll ")) == profile


def test_missing_profile_does_not_fall_back(familiar):
    with pytest.raises(ResourceNotFoundError) as caught:
        _run(familiar.get_agent_profile("ghost"))

    assert caught.value.details == {"resource": "agent_profile", "agent_alias": "ghost"}


def test_disabled_profile_is_rejected(familiar, store):
    store.save(AgentProfile(agent_id="sleeper", enabled=False))

    with pytest.raises(ProfileResolutionError) as caught:
        _run(familiar.get_agent_profile("sleeper"))

    assert caught.value.details == {"agent_alias": "sleeper", "reason": "profile_disabled"}


def test_invalid_profile_file_is_rejected_as_invalid_config(familiar, tmp_path):
    (tmp_path / "broken.md").write_text("---\nmodel_name: old\n---\n人设", encoding="utf-8")

    with pytest.raises(ProfileResolutionError) as caught:
        _run(familiar.get_agent_profile("broken"))

    assert caught.value.details == {"agent_alias": "broken", "reason": "invalid_config"}
