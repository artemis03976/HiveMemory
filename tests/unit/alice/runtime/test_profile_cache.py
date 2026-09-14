"""
AgentProfileCache 单元测试
"""

import pytest

from hivememory.alice.runtime.profile_cache import AgentProfileCache
from hivememory.core.models import ActorIdentity, AgentProfile
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity()
ISOLATED = make_workspace_identity(workspace_id="isolation_workspace")


def _actor(
    user_id: str = "u1",
    agent_id: str = "omni_doll",
    team_id: str | None = "team-a",
) -> ActorIdentity:
    return ActorIdentity(user_id=user_id, agent_id=agent_id, team_id=team_id)


def test_get_and_store_roundtrip():
    cache = AgentProfileCache()
    profile = AgentProfile(persona="coder persona")

    assert cache.get(MAIN, _actor(), "coder_doll") is None

    cache.store(MAIN, _actor(), "coder_doll", profile)

    assert cache.get(MAIN, _actor(), "coder_doll") is profile
    assert cache.size == 1


def test_lru_evicts_oldest_entry_and_counts():
    """容量满时按 LRU 淘汰最久未用条目，命中会刷新驻留顺序。"""
    cache = AgentProfileCache(max_size=2)
    first = AgentProfile(persona="first")
    second = AgentProfile(persona="second")
    third = AgentProfile(persona="third")

    cache.store(MAIN, _actor(), "doll_first", first)
    cache.store(MAIN, _actor(), "doll_second", second)
    # 命中 first 刷新其驻留顺序，使 second 成为最久未用条目。
    assert cache.get(MAIN, _actor(), "doll_first") is first
    cache.store(MAIN, _actor(), "doll_third", third)

    assert cache.get(MAIN, _actor(), "doll_second") is None
    assert cache.get(MAIN, _actor(), "doll_first") is first
    assert cache.get(MAIN, _actor(), "doll_third") is third
    assert cache.size == 2
    assert cache.evictions == 1
    assert cache.misses == 1


def test_hit_and_miss_counters():
    cache = AgentProfileCache()
    cache.store(MAIN, _actor(), "coder_doll", AgentProfile(persona="p"))

    assert cache.get(MAIN, _actor(), "coder_doll") is not None
    assert cache.get(MAIN, _actor(), "missing_doll") is None

    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_rejects_missing_scope_coordinates():
    """无 Workspace/Actor 坐标的 cache 读写不存在，缺失或错误坐标直接拒绝。"""
    cache = AgentProfileCache()

    with pytest.raises(TypeError):
        cache.get(None, _actor(), "coder_doll")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cache.store(MAIN, "not-actor", "coder_doll", AgentProfile(persona="p"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cache.key(MAIN, None, "coder_doll")  # type: ignore[arg-type]


def test_workspace_partitioning_at_cache_level():
    """同 Actor 同 alias 在不同 Workspace 分区各自缓存。"""
    cache = AgentProfileCache()
    main_profile = AgentProfile(persona="main")
    isolated_profile = AgentProfile(persona="isolated")

    cache.store(MAIN, _actor(), "coder_doll", main_profile)
    cache.store(ISOLATED, _actor(), "coder_doll", isolated_profile)

    assert cache.get(MAIN, _actor(), "coder_doll") is main_profile
    assert cache.get(ISOLATED, _actor(), "coder_doll") is isolated_profile


def test_clear_drops_all_entries():
    cache = AgentProfileCache()
    cache.store(MAIN, _actor(), "coder_doll", AgentProfile(persona="p"))
    cache.store(ISOLATED, _actor(), "coder_doll", AgentProfile(persona="q"))

    cache.clear()

    assert cache.size == 0
    assert cache.get(MAIN, _actor(), "coder_doll") is None
    assert cache.get(ISOLATED, _actor(), "coder_doll") is None
