"""
PendingAtom 内部存储层。

`_PendingAtomStore` 是子包私有的纯存储组件：独占 PendingAtom 字典与反查索引，
只提供原子级 CRUD 与索引维护，不感知状态机、不发事件。所有状态机校验与命令
组合由 `PendingAtomRuntime` 承担（见 runtime.py）。

设计依据：docs/mod/PendingAtomRuntimeDesign.md §4 / §6.2
"""

from __future__ import annotations

from typing import Dict, List, Optional

from hivememory.alice.runtime.models import PendingAtom
from hivememory.engines.generation.models import PendingAtomSettlement

from hivememory.alice.runtime.pending_atom.state import PendingAtomResolution


class _PendingAtomStore:
    """
    PendingAtom 纯存储层（子包私有）。

    独占 `_atoms` / `_intent_index` / `_canonical_index` 三份字典，
    以及 PR2 前暂留的 `_resolution` / `_redirects` 派生字段。
    仅负责存取与索引维护，不做状态机校验。
    """

    def __init__(self) -> None:
        self._atoms: Dict[str, PendingAtom] = {}
        self._intent_index: Dict[str, str] = {}
        self._redirects: Dict[str, PendingAtomSettlement] = {}
        self._canonical_index: Dict[str, List[str]] = {}
        self._resolution: Dict[str, PendingAtomResolution] = {}

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

    # ---- PR2 前暂留的派生字段（resolution / redirect） ----

    def set_resolution(self, alias: str, resolution: PendingAtomResolution) -> None:
        """记录 alias 的终结分类。"""
        self._resolution[alias] = resolution

    def get_resolution(self, alias: str) -> Optional[PendingAtomResolution]:
        """读取 alias 的终结分类。"""
        return self._resolution.get(alias)

    def set_redirect(self, alias: str, settlement: PendingAtomSettlement) -> None:
        """记录 alias 的 canonical redirect settlement。"""
        self._redirects[alias] = settlement

    def get_redirect(self, alias: str) -> Optional[PendingAtomSettlement]:
        """返回 alias 对应的 canonical redirect settlement。"""
        return self._redirects.get(alias)

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
        self._redirects.clear()
        self._canonical_index.clear()
        self._resolution.clear()

    @property
    def size(self) -> int:
        """当前存储的 pending atom 数量。"""
        return len(self._atoms)


__all__ = ["_PendingAtomStore"]
