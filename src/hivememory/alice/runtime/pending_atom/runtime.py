"""
PendingAtomRuntime - PendingAtom 全生命周期管理中心。

作为 Alice runtime 共享基础设施，主帧和子帧共享同一实例。
子帧 WRITE 后主帧自动可见，无需 merge。

外观 + 命令入口：所有 PendingAtom 的注册、结算、查询都通过本对象，
内部委托给子包私有的 `_PendingAtomStore` 做存取与索引维护。

设计依据：docs/mod/PendingAtomRuntimeDesign.md
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional
from uuid import uuid4

from hivememory.alice.runtime.models import PendingAtom, RuntimeScope
from hivememory.core.models import Identity
from hivememory.engines.generation.models import (
    PendingAtomSettlement,
    UpdateFocus,
    WriteFocus,
)

from hivememory.alice.runtime.pending_atom.state import (
    PendingAtomSnapshot,
    PendingAtomStatus,
)
from hivememory.alice.runtime.pending_atom.store import _PendingAtomStore

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 30) -> str:
    """将文本转为 alias 友好的 slug 片段。"""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len].rstrip("_")


class PendingAtomRuntime:
    """
    PendingAtom 全生命周期管理中心。

    所有 PendingAtom 的注册、状态迁移、结算、查询都通过本对象。
    资源所有权（_atoms、_intent_index、_canonical_index）由内部
    `_PendingAtomStore` 独占，外部不直接持有 store 实例。
    """

    def __init__(self) -> None:
        self._store = _PendingAtomStore()

    # ---- 命令（写入路径） ----

    def register_write(
        self,
        content: str,
        title: Optional[str],
        reason: Optional[str],
        identity: Identity,
        runtime_scope: Optional[RuntimeScope] = None,
    ) -> PendingAtom:
        """注册 WRITE pending atom，返回带有生成 alias 的 PendingAtom。status=PENDING。"""
        slug_source = title if title else content[:20]
        slug = _slugify(slug_source)
        if not slug:
            slug = "untitled"
        short_id = uuid4().hex[:4]
        pending_alias = f"draft_{slug}_{short_id}"
        intent_id = f"intent_{uuid4().hex[:12]}"
        focus = WriteFocus(
            content=content,
            reason=reason,
            title=title,
            identity=identity,
            pending_alias=pending_alias,
            intent_id=intent_id,
        )

        atom = PendingAtom(
            pending_alias=pending_alias,
            intent_id=intent_id,
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            focus=focus,
            identity=identity,
            runtime_scope=runtime_scope or RuntimeScope(),
        )
        self._store.put(atom)
        self._store.bind_intent(intent_id, pending_alias)
        logger.debug(f"Registered pending WRITE: {pending_alias} (intent={intent_id})")
        return atom

    def register_update(
        self,
        base_alias: str,
        base_uuid: str,
        instruction: str,
        content: Optional[str],
        identity: Identity,
        runtime_scope: Optional[RuntimeScope] = None,
    ) -> PendingAtom:
        """注册 UPDATE pending revision，返回带有生成 alias 的 PendingAtom。status=PENDING。"""
        short_id = uuid4().hex[:4]
        pending_alias = f"rev_{base_alias}_{short_id}"
        intent_id = f"intent_{uuid4().hex[:12]}"
        focus = UpdateFocus(
            instruction=instruction,
            content=content or None,
            base_alias=base_alias,
            base_uuid=base_uuid,
            identity=identity,
            pending_alias=pending_alias,
            intent_id=intent_id,
        )

        atom = PendingAtom(
            pending_alias=pending_alias,
            intent_id=intent_id,
            status=PendingAtomStatus.PENDING,
            source_verb="UPDATE",
            focus=focus,
            identity=identity,
            runtime_scope=runtime_scope or RuntimeScope(),
        )
        self._store.put(atom)
        self._store.bind_intent(intent_id, pending_alias)
        logger.debug(f"Registered pending UPDATE: {pending_alias} (intent={intent_id})")
        return atom

    def settle(self, settlement: PendingAtomSettlement) -> None:
        """
        Apply a settlement from the generation pipeline to update pending atom state.

        Args:
            settlement: PendingAtomSettlement instance
        """
        pending_alias = settlement.pending_alias
        atom = self._store.get(pending_alias)

        if atom is None and settlement.intent_id:
            atom = self._store.get_by_intent(settlement.intent_id)

        if atom is None:
            logger.warning(
                f"Settlement for unknown pending atom: "
                f"alias={settlement.pending_alias}, intent={settlement.intent_id}"
            )
            return

        atom.status = PendingAtomStatus.SETTLED
        self._store.set_resolution(atom.pending_alias, settlement.resolution)

        atom.settlement = settlement
        if settlement.canonical_alias or settlement.canonical_uuid:
            self._store.set_redirect(atom.pending_alias, settlement)
            if settlement.canonical_uuid:
                self._store.bind_canonical(
                    settlement.canonical_uuid,
                    atom.pending_alias,
                )

        logger.debug(
            f"Settlement applied to '{atom.pending_alias}': "
            f"resolution={settlement.resolution.value}, "
            f"canonical={settlement.canonical_alias}"
        )

    # ---- 查询（读取路径） ----

    def get(self, pending_alias: str) -> Optional[PendingAtom]:
        """返回 PendingAtom 原始引用（仅用于持有 focus / runtime_scope 等数据）。"""
        return self._store.get(pending_alias)

    def get_by_intent_id(self, intent_id: str) -> Optional[PendingAtom]:
        """通过 intent_id 查询 pending atom。"""
        return self._store.get_by_intent(intent_id)

    def get_redirect(self, pending_alias: str) -> Optional[PendingAtomSettlement]:
        """返回 pending alias 对应的 canonical redirect settlement。"""
        return self._store.get_redirect(pending_alias)

    def get_pending_aliases_for_canonical_uuid(self, canonical_uuid: str) -> List[str]:
        """返回指向同一 canonical UUID 的 pending alias 列表。"""
        return self._store.aliases_by_canonical(canonical_uuid)

    def has(self, alias: str) -> bool:
        """检查 alias 是否为已注册的 pending atom。"""
        return self._store.has(alias)

    def all_aliases(self) -> List[str]:
        """返回所有已注册的 pending alias。"""
        return self._store.all_aliases()

    def all_atoms(self) -> List[PendingAtom]:
        """返回所有已注册的 PendingAtom。"""
        return self._store.all_atoms()

    def snapshot(self, pending_alias: str) -> Optional[PendingAtomSnapshot]:
        """
        返回 pending alias 对应的统一状态视图。

        派生路径：
        - 已结算（resolution 命中）：status=SETTLED + 对应 resolution
        - 未结算：直接读取 ``atom.status``（来自统一 ``PendingAtomStatus``）
        """
        atom = self._store.get(pending_alias)
        if atom is None:
            return None

        resolution = self._store.get_resolution(pending_alias)
        settlement = self._store.get_redirect(pending_alias)

        if resolution is not None:
            canonical_alias = settlement.canonical_alias if settlement else None
            canonical_uuid = settlement.canonical_uuid if settlement else None
            if not resolution.has_canonical:
                canonical_alias = None
                canonical_uuid = None
            return PendingAtomSnapshot(
                pending_alias=pending_alias,
                status=PendingAtomStatus.SETTLED,
                resolution=resolution,
                canonical_alias=canonical_alias,
                canonical_uuid=canonical_uuid,
            )

        return PendingAtomSnapshot(
            pending_alias=pending_alias,
            status=atom.status,
            resolution=None,
            canonical_alias=None,
            canonical_uuid=None,
        )

    # ---- 生命周期 ----

    def clear(self) -> None:
        """清空全部 pending atom。"""
        self._store.clear()

    @property
    def size(self) -> int:
        """当前缓存的 pending atom 数量。"""
        return self._store.size


__all__ = ["PendingAtomRuntime"]
