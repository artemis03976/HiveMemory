"""Workspace 共享的 Profile 解析结果缓存（ProfileCache，A2 §2.3 / §3）。

迁移源为 Alice 侧 ``alice/runtime/profile_cache.py``（``AgentProfileCache``）。
旧实现按完整 Actor 授权坐标分区并缓存"已通过授权的裸 AgentProfile"；这里
改为按 ``(Workspace, agent_alias)`` 寻址，条目随存源原子的 policy 依据，命中
时由 resolver 对当前 Actor 逐次授权，不缓存任何 Actor 的授权结论。

- 条目只来自 atom 来源的解析结果；builtin Profile 无源原子，不进缓存；
- 反向索引 ``(Workspace, source_memory_id) → agent_alias`` 供失效事件按
  ``memory_id`` 定位条目（事件载荷不含 alias）；
- 保存与读取均复制 ``AgentProfile``；全局 LRU 容量。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from uuid import UUID

from hivememory.core.models import AgentProfile, MemoryAccessPolicy, WorkspaceIdentity
from hivememory.workspace.cache.keys import AtomIdKey, ProfileKey


@dataclass(frozen=True, slots=True)
class ProfileCacheEntry:
    """一条 Profile 解析结果：能力描述 + 源原子的授权依据与关联。

    ``access_policy`` 是命中授权的唯一依据；``source_memory_id`` 用于失效
    对账；``source_version`` 仅供诊断，不作为当前性依据。
    """

    profile: AgentProfile
    access_policy: MemoryAccessPolicy
    source_memory_id: UUID
    source_version: int

    def isolated(self) -> ProfileCacheEntry:
        """返回可变嵌套对象均为独立副本的条目。"""
        return replace(
            self,
            profile=self.profile.model_copy(deep=True),
            access_policy=self.access_policy.model_copy(deep=True),
        )


class ProfileCache:
    """按 ``(Workspace, agent_alias)`` 寻址的 Profile 解析结果 LRU 缓存。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("ProfileCache capacity 必须至少为 1")
        self._capacity = capacity
        self._entries: OrderedDict[ProfileKey, ProfileCacheEntry] = OrderedDict()
        self._source_to_alias: dict[AtomIdKey, str] = {}
        # 可观测统计：命中/未命中与容量淘汰次数
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, workspace: WorkspaceIdentity, agent_alias: str) -> ProfileCacheEntry | None:
        """读取条目副本，未命中返回 None。"""
        key = ProfileKey(workspace, agent_alias)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.isolated()

    def put(
        self,
        workspace: WorkspaceIdentity,
        agent_alias: str,
        entry: ProfileCacheEntry,
    ) -> None:
        """保存条目副本并登记源原子反向索引；容量满时按 LRU 淘汰。"""
        key = ProfileKey(workspace, agent_alias)
        self._remove(key)
        # 同一源原子若以其他 alias 缓存过（如 alias 变更后尚未失效），以本次为准。
        source_key = AtomIdKey(workspace, entry.source_memory_id)
        stale_alias = self._source_to_alias.get(source_key)
        if stale_alias is not None:
            self._remove(ProfileKey(workspace, stale_alias))
        self._entries[key] = entry.isolated()
        self._source_to_alias[source_key] = agent_alias
        while len(self._entries) > self._capacity:
            evicted_key, evicted_entry = self._entries.popitem(last=False)
            self._drop_source_link(evicted_key, evicted_entry)
            self._evictions += 1

    def evict_source(self, workspace: WorkspaceIdentity, memory_id: UUID) -> bool:
        """移除由指定源原子解析出的条目；返回是否确有条目被移除。"""
        alias = self._source_to_alias.get(AtomIdKey(workspace, memory_id))
        if alias is None:
            return False
        return self._remove(ProfileKey(workspace, alias))

    def clear(self) -> int:
        """清空全部条目与索引（统计计数保持累计），返回清理的条目数。"""
        size = len(self._entries)
        self._entries.clear()
        self._source_to_alias.clear()
        return size

    def _remove(self, key: ProfileKey) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._drop_source_link(key, entry)
        return True

    def _drop_source_link(self, key: ProfileKey, entry: ProfileCacheEntry) -> None:
        source_key = AtomIdKey(key.workspace, entry.source_memory_id)
        if self._source_to_alias.get(source_key) == key.agent_alias:
            del self._source_to_alias[source_key]

    @property
    def size(self) -> int:
        """当前缓存的条目数量。"""
        return len(self._entries)

    @property
    def hits(self) -> int:
        """读取路径的累计命中次数。"""
        return self._hits

    @property
    def misses(self) -> int:
        """读取路径的累计未命中次数。"""
        return self._misses

    @property
    def evictions(self) -> int:
        """LRU 容量淘汰的累计次数。"""
        return self._evictions


__all__ = ["ProfileCache", "ProfileCacheEntry"]
