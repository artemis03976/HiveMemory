"""
PendingAtom 内部存储层。

`_PendingAtomStore` 是子包私有的纯存储组件：独占 PendingAtom 字典与反查索引，
只提供原子级 CRUD 与索引维护，不感知状态机、不发事件。所有状态机校验与命令
组合由 `PendingAtomRuntime` 承担（见 runtime.py）。

设计依据：docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md §4 / §6.2
"""

from __future__ import annotations

from typing import Dict, List, Optional

from hivememory.core.models.pending import (
    PendingAtom,
)


class _PendingAtomStore:
    """
    PendingAtom 纯存储层（子包私有）。

    独占 `_atoms` / `_intent_index` / `_canonical_index` 三份字典。
    仅负责存取与索引维护，不做状态机校验。
    """

    def __init__(self) -> None:
        self._atoms: Dict[str, PendingAtom] = {}
        self._intent_index: Dict[str, str] = {}
        self._canonical_index: Dict[str, List[str]] = {}

    # ---- 原子存取 ----

    def put(self, atom: PendingAtom) -> None:
        """写入 PendingAtom。"""
        self._atoms[atom.pending_alias] = atom

    def get(self, alias: str) -> Optional[PendingAtom]:
        """通过 pending alias 查询。"""
        return self._atoms.get(alias)

    def has(self, alias: str) -> bool:
        """检查 alias 是否为已注册的 pending atom。"""
        return alias in self._atoms

    def get_by_intent(self, intent_id: str) -> Optional[PendingAtom]:
        """通过 intent_id 查询 pending atom。"""
        alias = self._intent_index.get(intent_id)
        if alias:
            return self._atoms.get(alias)
        return None

    # ---- 索引维护 ----

    def bind_intent(self, intent_id: str, alias: str) -> None:
        """登记 intent_id -> pending_alias 反查索引。"""
        self._intent_index[intent_id] = alias

    def bind_canonical(self, canonical_uuid: str, alias: str) -> None:
        """登记 canonical_uuid -> [pending_alias] 反查索引。"""
        aliases = self._canonical_index.setdefault(canonical_uuid, [])
        if alias not in aliases:
            aliases.append(alias)

    def aliases_by_canonical(self, canonical_uuid: str) -> List[str]:
        """返回指向同一 canonical UUID 的 pending alias 列表。"""
        return list(self._canonical_index.get(canonical_uuid, []))

    def delete(self, alias: str) -> None:
        """删除 atom 及其所有索引。"""
        atom = self._atoms.pop(alias, None)
        if atom is None:
            return

        self._intent_index.pop(atom.intent_id, None)

        if atom.settlement and atom.settlement.canonical_uuid:
            aliases = self._canonical_index.get(atom.settlement.canonical_uuid, [])
            if alias in aliases:
                aliases.remove(alias)
            if not aliases:
                self._canonical_index.pop(atom.settlement.canonical_uuid, None)

    # ---- 集合视图 ----

    def all_aliases(self) -> List[str]:
        """返回所有已注册的 pending alias。"""
        return list(self._atoms.keys())

    def all_atoms(self) -> List[PendingAtom]:
        """返回所有已注册的 PendingAtom。"""
        return list(self._atoms.values())

    def clear(self) -> None:
        """清空全部存储与索引。"""
        self._atoms.clear()
        self._intent_index.clear()
        self._canonical_index.clear()

    @property
    def size(self) -> int:
        """当前存储的 pending atom 数量。"""
        return len(self._atoms)


__all__ = ["_PendingAtomStore"]
