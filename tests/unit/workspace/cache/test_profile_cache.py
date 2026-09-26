"""ProfileCache 的单元测试（A2 §2.3 Profile 解析结果缓存）。

被测对象：``(Workspace, agent_alias)`` 寻址、源原子反向索引（失效事件按
memory_id 定位条目）与 LRU 淘汰的一致性，以及条目的引用隔离。
"""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from hivememory.core.models import AgentProfile, MemoryAccessPolicy, MemoryVisibility
from hivememory.workspace.cache.profile import ProfileCache, ProfileCacheEntry
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


def _entry(agent_id: str, *, source_memory_id=None) -> ProfileCacheEntry:
    return ProfileCacheEntry(
        profile=AgentProfile(agent_id=agent_id, persona=f"{agent_id} persona"),
        access_policy=MemoryAccessPolicy.public(),
        source_memory_id=source_memory_id or uuid4(),
        source_version=1,
    )


def test_evict_source_removes_entry_resolved_from_that_atom():
    """按源原子 memory_id 失效对应条目，其他条目不受影响。

    捕获失效事件只携带 memory_id 时无法定位 Profile 条目的缺陷。
    """
    cache = ProfileCache(capacity=8)
    coder = _entry("coder_doll")
    reviewer = _entry("reviewer_doll")
    cache.put(MAIN, "coder_doll", coder)
    cache.put(MAIN, "reviewer_doll", reviewer)

    assert cache.evict_source(MAIN, coder.source_memory_id) is True

    assert cache.get(MAIN, "coder_doll") is None
    assert cache.get(MAIN, "reviewer_doll").profile.agent_id == "reviewer_doll"


def test_same_source_cached_under_new_alias_replaces_old_alias_entry():
    """同一源原子以新 alias 缓存时，旧 alias 条目随之移除，不留下可返回的旧值。"""
    cache = ProfileCache(capacity=8)
    source_id = uuid4()
    cache.put(MAIN, "old_alias", _entry("old_alias", source_memory_id=source_id))

    cache.put(MAIN, "new_alias", _entry("new_alias", source_memory_id=source_id))

    assert cache.get(MAIN, "old_alias") is None
    assert cache.evict_source(MAIN, source_id) is True
    assert cache.get(MAIN, "new_alias") is None


def test_lru_eviction_keeps_source_index_consistent():
    """容量淘汰后，被淘汰条目的源原子失效不再命中任何条目。"""
    cache = ProfileCache(capacity=1)
    evicted = _entry("first_doll")
    cache.put(MAIN, "first_doll", evicted)
    cache.put(MAIN, "second_doll", _entry("second_doll"))

    assert cache.evict_source(MAIN, evicted.source_memory_id) is False
    assert cache.get(MAIN, "second_doll").profile.agent_id == "second_doll"


def test_returned_entry_is_isolated_from_cache():
    """修改读取到的 Profile 与 policy 不影响后续读取与授权依据。"""
    cache = ProfileCache(capacity=8)
    private = MemoryAccessPolicy(visibility=MemoryVisibility.PRIVATE, target_agent_id="a1")
    cache.put(MAIN, "coder_doll", replace(_entry("coder_doll"), access_policy=private))

    first = cache.get(MAIN, "coder_doll")
    first.profile.persona = "mutated"
    first.access_policy.target_agent_id = "someone_else"

    again = cache.get(MAIN, "coder_doll")
    assert (again.profile.persona, again.access_policy.target_agent_id) == (
        "coder_doll persona",
        "a1",
    )
