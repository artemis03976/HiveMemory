"""Workspace 共享的完整原子缓存（AtomCache）。

缓存由读取视图持有，唯一读者是记忆读取器；记忆库对缓存的
唯一接触是出向的失效事件，库不读取也不写入缓存。

本模块只提供存取、索引与容量淘汰，不做授权、不接收全局路由、不自行
回源：交付边界的逐次授权、epoch 守护回填与冷读都由 resolver 负责。

- 主键 ``(Workspace, memory_id)``，alias 正向索引 ``(Workspace, alias) →
  memory_id`` 与反向索引 ``(Workspace, memory_id) → alias`` 同步维护，
  失效一条原子时连同其 alias 一并移除（含被释放的旧 alias）；
- 保存与读取均复制完整 ``MemoryAtom``（copy-on-read），调用方修改返回值
  不会改变缓存或后续授权结果；
- 全局 LRU 容量，不承诺 Workspace 独立配额。
"""

from __future__ import annotations

from collections import OrderedDict
from uuid import UUID

from hivememory.core.models import MemoryAtom, WorkspaceIdentity
from hivememory.workspace.cache.keys import AtomAliasKey, AtomIdKey


class AtomCache:
    """按 Workspace 与资源寻址的完整原子 LRU 缓存。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("AtomCache capacity 必须至少为 1")
        self._capacity = capacity
        self._atoms: OrderedDict[AtomIdKey, MemoryAtom] = OrderedDict()
        self._alias_to_id: dict[AtomAliasKey, UUID] = {}
        self._id_to_alias: dict[AtomIdKey, str] = {}

    def get_by_id(self, workspace: WorkspaceIdentity, memory_id: UUID) -> MemoryAtom | None:
        """按 Workspace 分区内的 UUID 读取原子副本，未命中返回 None。"""
        key = AtomIdKey(workspace, memory_id)
        atom = self._atoms.get(key)
        if atom is None:
            return None
        self._atoms.move_to_end(key)
        return atom.model_copy(deep=True)

    def get_by_alias(self, workspace: WorkspaceIdentity, alias: str) -> MemoryAtom | None:
        """按 Workspace 分区内的 alias 读取原子副本，未命中返回 None。"""
        memory_id = self._alias_to_id.get(AtomAliasKey(workspace, alias))
        if memory_id is None:
            return None
        return self.get_by_id(workspace, memory_id)

    def put(self, atom: MemoryAtom) -> None:
        """保存原子副本并以其当前 alias 建立索引；分区取原子自身的 Workspace 归属。

        同一 memory_id 再次写入时替换旧值与旧 alias 索引；alias 已指向另一
        memory_id 时改指当前原子，被取代者只保留 UUID 寻址。
        """
        workspace = atom.workspace_identity
        key = AtomIdKey(workspace, atom.id)
        self._unlink_alias(key)
        self._atoms[key] = atom.model_copy(deep=True)
        self._atoms.move_to_end(key)
        alias = atom.index.alias
        if alias:
            alias_key = AtomAliasKey(workspace, alias)
            previous_id = self._alias_to_id.get(alias_key)
            if previous_id is not None and previous_id != atom.id:
                self._id_to_alias.pop(AtomIdKey(workspace, previous_id), None)
            self._alias_to_id[alias_key] = atom.id
            self._id_to_alias[key] = alias
        while len(self._atoms) > self._capacity:
            evicted_key, _ = self._atoms.popitem(last=False)
            self._unlink_alias(evicted_key)

    def evict(self, workspace: WorkspaceIdentity, memory_id: UUID) -> bool:
        """移除一条原子及其 alias 索引；返回是否确有条目被移除。"""
        key = AtomIdKey(workspace, memory_id)
        self._unlink_alias(key)
        return self._atoms.pop(key, None) is not None

    def clear(self) -> int:
        """清空全部条目与索引，返回清理的条目数。"""
        size = len(self._atoms)
        self._atoms.clear()
        self._alias_to_id.clear()
        self._id_to_alias.clear()
        return size

    def _unlink_alias(self, key: AtomIdKey) -> None:
        alias = self._id_to_alias.pop(key, None)
        if alias is None:
            return
        alias_key = AtomAliasKey(key.workspace, alias)
        if self._alias_to_id.get(alias_key) == key.memory_id:
            del self._alias_to_id[alias_key]

    @property
    def size(self) -> int:
        """当前缓存的原子数量。"""
        return len(self._atoms)


__all__ = ["AtomCache"]
