"""
PendingAtomRuntime - PendingAtom 全生命周期管理中心。

作为 Alice runtime 共享基础设施，主帧和子帧共享同一实例。
子帧 WRITE 后主帧自动可见，无需 merge。

外观 + 命令入口：所有 PendingAtom 的注册、结算、查询都通过本对象，
内部委托给子包私有的 `_PendingAtomStore` 做存取与索引维护。

设计依据：docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md
"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from hivememory.agent_runtime.pending_atom.store import _PendingAtomStore
from hivememory.core.models import Identity
from hivememory.core.models.pending import (
    InvalidStateTransition,
    PendingAtom,
    PendingAtomMaterializeTask,
    PendingAtomSettlement,
    PendingAtomSnapshot,
    PendingAtomStatus,
    RuntimeScope,
    UpdateFocus,
    WriteFocus,
    is_legal_transition,
)

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

    def _set_status(self, atom: PendingAtom, target: PendingAtomStatus) -> None:
        """执行一次合法的 PendingAtom 生命周期迁移，非法迁移抛出异常。"""
        if not is_legal_transition(atom.status, target):
            raise InvalidStateTransition(
                f"PendingAtom '{atom.pending_alias}': "
                f"{atom.status.value} -> {target.value} is not a legal transition"
            )
        atom.status = target

    # ---- 命令（写入路径） ----

    def register_write(
        self,
        content: str,
        title: str | None,
        reason: str | None,
        identity: Identity,
        runtime_scope: RuntimeScope,
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
        )

        atom = PendingAtom(
            pending_alias=pending_alias,
            intent_id=intent_id,
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            focus=focus,
            identity=identity,
            runtime_scope=runtime_scope,
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
        content: str | None,
        identity: Identity,
        runtime_scope: RuntimeScope,
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
        )

        atom = PendingAtom(
            pending_alias=pending_alias,
            intent_id=intent_id,
            status=PendingAtomStatus.PENDING,
            source_verb="UPDATE",
            focus=focus,
            identity=identity,
            runtime_scope=runtime_scope,
        )
        self._store.put(atom)
        self._store.bind_intent(intent_id, pending_alias)
        logger.debug(f"Registered pending UPDATE: {pending_alias} (intent={intent_id})")
        return atom

    def mark_failed(self, pending_alias: str) -> None:
        """将 MATERIALIZING 的 atom 迁移到 FAILED（幂等：非 MATERIALIZING 时静默跳过）。"""
        atom = self._store.get(pending_alias)
        if atom is None or atom.status != PendingAtomStatus.MATERIALIZING:
            logger.warning(
                f"mark_failed() skipped: alias={pending_alias}, "
                f"status={atom.status.value if atom else 'not found'}"
            )
            return
        self._set_status(atom, PendingAtomStatus.FAILED)

    def cancel(self, pending_alias: str, reason: str | None = None) -> None:
        """把 in-flight 状态的 atom 迁移到 CANCELLED。"""
        atom = self._store.get(pending_alias)
        if atom is None:
            logger.warning(f"cancel() skipped: alias={pending_alias}, status=not found")
            return
        if atom.status not in {
            PendingAtomStatus.PENDING,
            PendingAtomStatus.MATERIALIZING,
        }:
            logger.warning(f"cancel() skipped: alias={pending_alias}, status={atom.status.value}")
            return
        self._set_status(atom, PendingAtomStatus.CANCELLED)

    def cancel_run(self, run_id: str, reason: str | None = None) -> list[str]:
        """取消一个 run 产生的全部 in-flight PendingAtom。"""
        cancelled: list[str] = []
        for atom in self._store.all_atoms():
            if atom.runtime_scope.run_id != run_id:
                continue
            if atom.status not in {
                PendingAtomStatus.PENDING,
                PendingAtomStatus.MATERIALIZING,
            }:
                continue
            self.cancel(atom.pending_alias, reason=reason)
            cancelled.append(atom.pending_alias)
        return cancelled

    def cancel_frame(self, frame_id: str, reason: str | None = None) -> list[str]:
        """取消一个执行 frame 产生的全部 in-flight PendingAtom。"""
        cancelled: list[str] = []
        for atom in self._store.all_atoms():
            if atom.runtime_scope.frame_id != frame_id:
                continue
            if atom.status not in {
                PendingAtomStatus.PENDING,
                PendingAtomStatus.MATERIALIZING,
            }:
                continue
            self.cancel(atom.pending_alias, reason=reason)
            cancelled.append(atom.pending_alias)
        return cancelled

    def expire(self, pending_alias: str) -> None:
        """把 PENDING 状态的 atom 迁移到 EXPIRED。"""
        atom = self._store.get(pending_alias)
        if atom is None:
            logger.warning(f"expire() skipped: alias={pending_alias}, status=not found")
            return
        self._set_status(atom, PendingAtomStatus.EXPIRED)

    def start_materializing(self, pending_alias: str) -> None:
        """把 PENDING 状态的 atom 迁移到 MATERIALIZING。"""
        atom = self._store.get(pending_alias)
        if atom is None:
            logger.warning(
                f"start_materializing() skipped: alias={pending_alias}, status=not found"
            )
            return
        self._set_status(atom, PendingAtomStatus.MATERIALIZING)

    def claim_for_materialization(self, aliases: list[str]) -> list[PendingAtomMaterializeTask]:
        """将 PENDING 的 atom 迁移到 MATERIALIZING 并返回 Task 投影。非 PENDING 的静默跳过（幂等）。"""
        tasks = []
        for alias in aliases:
            atom = self._store.get(alias)
            if atom is None or atom.status != PendingAtomStatus.PENDING:
                continue
            self.start_materializing(alias)
            tasks.append(PendingAtomMaterializeTask.from_pending_atom(atom))
        return tasks

    def settle(self, settlement: PendingAtomSettlement) -> None:
        """应用来自生成管线的一次 settlement，更新 pending atom 状态。

        settlement 的 intent 必须与原 PendingAtom 匹配（见 docs/alice/pending-atom.md §8），
        匹配失败只记录 warning 并跳过，不破坏既有状态。

        Args:
            settlement: PendingAtomSettlement 结算视图
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

        # 校验 intent_id 是否匹配
        if atom.intent_id != settlement.intent_id:
            logger.warning(
                f"settle() intent mismatch: alias={atom.pending_alias}, "
                f"atom_intent={atom.intent_id}, settlement_intent={settlement.intent_id}, skipping"
            )
            return

        self._set_status(atom, PendingAtomStatus.SETTLED)

        atom.settlement = settlement
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

    def get(self, pending_alias: str) -> PendingAtom | None:
        """返回 PendingAtom 原始引用（仅用于持有 focus / runtime_scope 等数据）。"""
        return self._store.get(pending_alias)

    def get_by_intent_id(self, intent_id: str) -> PendingAtom | None:
        """通过 intent_id 查询 pending atom。"""
        return self._store.get_by_intent(intent_id)

    def get_redirect(self, pending_alias: str) -> PendingAtomSettlement | None:
        """返回 pending alias 对应的结算视图。

        兼容旧调用名；redirect 信息现在直接从 ``PendingAtom.settlement`` 派生。
        """
        atom = self._store.get(pending_alias)
        if atom is None:
            return None
        return atom.settlement

    def get_pending_aliases_for_canonical_uuid(self, canonical_uuid: str) -> list[str]:
        """返回指向同一 canonical UUID 的 pending alias 列表。"""
        return self._store.aliases_by_canonical(canonical_uuid)

    def has(self, alias: str) -> bool:
        """检查 alias 是否为已注册的 pending atom。"""
        return self._store.has(alias)

    def all_aliases(self) -> list[str]:
        """返回所有已注册的 pending alias。"""
        return self._store.all_aliases()

    def all_atoms(self) -> list[PendingAtom]:
        """返回所有已注册的 PendingAtom。"""
        return self._store.all_atoms()

    def aliases_by_frame(self, frame_id: str) -> list[str]:
        """返回一个执行 frame 产生的 pending alias 列表。"""
        return [
            atom.pending_alias
            for atom in self._store.all_atoms()
            if atom.runtime_scope.frame_id == frame_id
        ]

    def pending_aliases_by_run(self, run_id: str) -> list[str]:
        """返回一个 run 内仍为 PENDING 的可物化 alias 列表。"""
        return [
            atom.pending_alias
            for atom in self._store.all_atoms()
            if atom.runtime_scope.run_id == run_id and atom.status == PendingAtomStatus.PENDING
        ]

    def snapshot(self, pending_alias: str) -> PendingAtomSnapshot | None:
        """
        返回 pending alias 对应的统一状态视图。

        派生路径：
        - 已结算：从 ``atom.settlement`` 读取 resolution / canonical refs
        - 未结算：直接读取 ``atom.status``（来自统一 ``PendingAtomStatus``）
        """
        atom = self._store.get(pending_alias)
        if atom is None:
            return None

        settlement = atom.settlement
        if atom.status == PendingAtomStatus.SETTLED and settlement is not None:
            resolution = settlement.resolution
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

    def tasks_by_run(self, run_id: str) -> list[PendingAtomMaterializeTask]:
        """返回本 run 产生的全部 PendingAtom 的不可变物化请求投影。

        父帧与子帧共用同一 run_id（RuntimeScope.run_id），因此无需额外合并。
        """
        return [
            PendingAtomMaterializeTask.from_pending_atom(atom)
            for atom in self._store.all_atoms()
            if atom.runtime_scope.run_id == run_id
        ]

    def evict_by_run(self, current_run_id: str) -> None:
        """双步清理：删除既有 EXPIRED，再将上轮已完成 atom → EXPIRED。

        在 collect_tasks_by_run 之后调用（新 run 开始时），确保上轮的
        SETTLED/FAILED/CANCELLED atom 在本轮结算后保留一轮过期提示窗口。
        """
        # 步骤一：删除上一个回收周期已标记的 EXPIRED。
        for alias in [
            a
            for a in self._store.all_aliases()
            if self._store.get(a).status == PendingAtomStatus.EXPIRED
        ]:
            self._store.delete(alias)
            logger.debug(f"evict_by_run: deleted EXPIRED atom '{alias}'")

        # 步骤二：将属于旧 run 且已离开 in-flight 的 atom 迁移到 EXPIRED。
        # 新迁移的 EXPIRED 会保留到下一次 evict_by_run，供 resolver 返回 expired。
        for atom in self._store.all_atoms():
            if atom.runtime_scope.run_id == current_run_id:
                continue
            if atom.status in {
                PendingAtomStatus.SETTLED,
                PendingAtomStatus.FAILED,
                PendingAtomStatus.CANCELLED,
            }:
                self._set_status(atom, PendingAtomStatus.EXPIRED)
                self._store.put(atom)

    # ---- 生命周期 ----

    def clear(self) -> None:
        """清空全部 pending atom。"""
        self._store.clear()

    @property
    def size(self) -> int:
        """当前缓存的 pending atom 数量。"""
        return self._store.size


__all__ = ["PendingAtomRuntime"]
