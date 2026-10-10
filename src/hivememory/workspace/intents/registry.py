"""进程级写入意图登记、认领与结算；只保存归属、发起者和进程关联。"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from hivememory.components.bus.async_bus import AsyncSystemBus
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.core.models.pending import (
    PendingAtom,
    PendingAtomMaterializeTask,
    PendingAtomSettlement,
    PendingAtomStatus,
    UpdateFocus,
    WriteFocus,
    is_legal_transition,
)

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 30) -> str:
    """沿用 MTP 的 alias 片段规则，空片段统一回退为 untitled。"""
    slug = re.sub(r"[^a-z0-9\s_]", "", text.lower().strip())
    slug = re.sub(r"\s+", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_")[:max_len].rstrip("_") or "untitled"


class WriteIntentRegistry:
    """全 Workspace 可回读的登记，同步操作在单个事件循环内原子完成。

    ``process_id`` 仅供认领与取消关联，不限制回读。结算后句柄保留到重启，
    不产生 EXPIRED，也不回填 canonical 缓存。
    """

    def __init__(self) -> None:
        self._atoms: dict[str, PendingAtom] = {}
        self._intent_aliases: dict[str, str] = {}
        self._bus: AsyncSystemBus | None = None

    def register_write(
        self,
        focus: WriteFocus,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None,
    ) -> PendingAtom:
        """登记 WRITE；返回独立副本，调用方不能改写登记中的状态。"""
        prefix = f"draft_{_slugify(focus.title or focus.content[:20])}"
        return self._register(prefix, focus, belong_to, from_actor, process_id)

    def register_update(
        self,
        focus: UpdateFocus,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None,
    ) -> PendingAtom:
        """登记 UPDATE；基础原子的可读性由能力层先行验证。"""
        return self._register(f"rev_{focus.base_alias}", focus, belong_to, from_actor, process_id)

    def _register(
        self,
        prefix: str,
        focus: WriteFocus | UpdateFocus,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None,
    ) -> PendingAtom:
        # alias 永不复用；短后缀冲突时重新生成，不能覆盖已结算的旧句柄。
        alias = f"{prefix}_{uuid4().hex[:4]}"
        while alias in self._atoms:
            alias = f"{prefix}_{uuid4().hex[:4]}"
        intent_id = f"intent_{uuid4().hex[:12]}"
        while intent_id in self._intent_aliases:
            intent_id = f"intent_{uuid4().hex[:12]}"
        atom = PendingAtom(
            pending_alias=alias,
            intent_id=intent_id,
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE" if isinstance(focus, WriteFocus) else "UPDATE",
            focus=focus.model_copy(deep=True),
            belong_to=belong_to,
            from_actor=from_actor,
            process_id=process_id,
        )
        self._atoms[alias] = atom
        self._intent_aliases[intent_id] = alias
        return atom.model_copy(deep=True)

    def get(self, alias: str, workspace_identity: WorkspaceIdentity) -> PendingAtom | None:
        """只交付目标 Workspace 的记录；发起者和进程均不参与授权。"""
        atom = self._atoms.get(alias)
        if atom is None or atom.belong_to != workspace_identity:
            return None
        return atom.model_copy(deep=True)

    def claim_process(self, process_id: str) -> list[PendingAtomMaterializeTask]:
        """认领指定进程尚为 PENDING 的记录；重复认领不重复派发。"""
        tasks: list[PendingAtomMaterializeTask] = []
        for atom in self._atoms.values():
            if atom.process_id == process_id and atom.status == PendingAtomStatus.PENDING:
                task = PendingAtomMaterializeTask.from_pending_atom(atom)
                atom.status = PendingAtomStatus.MATERIALIZING
                tasks.append(task)
        return tasks

    def cancel_process(self, process_id: str) -> list[str]:
        """关闭进程时只取消未认领意图，不改变已交给物化流水线的记录。"""
        aliases: list[str] = []
        for atom in self._atoms.values():
            if atom.process_id == process_id and atom.status == PendingAtomStatus.PENDING:
                atom.status = PendingAtomStatus.CANCELLED
                aliases.append(atom.pending_alias)
        return aliases

    def cancel_aliases(self, aliases: list[str], *, process_id: str) -> list[str]:
        """撤回指定进程仍为 PENDING 的意图；其他进程的句柄与已认领记录不受影响。"""
        cancelled: list[str] = []
        for alias in dict.fromkeys(aliases):
            atom = self._atoms.get(alias)
            if (
                atom is not None
                and atom.process_id == process_id
                and atom.status == PendingAtomStatus.PENDING
            ):
                atom.status = PendingAtomStatus.CANCELLED
                cancelled.append(alias)
        return cancelled

    def subscribe(self, bus: AsyncSystemBus) -> None:
        """订阅既有结算事件；重复装配幂等，切换总线时先取消旧订阅。"""
        if self._bus is bus:
            return
        self.unsubscribe()
        self._bus = bus
        bus.subscribe(GlobalEvents.PENDING_ATOM_SETTLED, self.on_settled)
        bus.subscribe(GlobalEvents.PENDING_ATOM_FAILED, self.on_failed)
        bus.subscribe(GlobalEvents.PENDING_ATOM_CANCELLED, self.on_cancelled)

    def unsubscribe(self) -> None:
        """解除订阅，不删除已经登记或结算的句柄。"""
        if self._bus is None:
            return
        self._bus.unsubscribe(GlobalEvents.PENDING_ATOM_SETTLED, self.on_settled)
        self._bus.unsubscribe(GlobalEvents.PENDING_ATOM_FAILED, self.on_failed)
        self._bus.unsubscribe(GlobalEvents.PENDING_ATOM_CANCELLED, self.on_cancelled)
        self._bus = None

    async def on_settled(self, *, settlement: PendingAtomSettlement) -> None:
        """严格匹配 intent_id；过时或重复的结算不逆转既有终态。"""
        atom = self._matching(settlement.pending_alias, settlement.intent_id)
        if atom is None or not is_legal_transition(atom.status, PendingAtomStatus.SETTLED):
            return
        atom.settlement = settlement.model_copy(deep=True)
        atom.status = PendingAtomStatus.SETTLED

    async def on_failed(self, *, pending_alias: str, intent_id: str | None = None) -> None:
        """应用失败事件；旧载荷只有不可重用 alias，提供 intent_id 时追加匹配。"""
        atom = self._matching(pending_alias, intent_id)
        if atom is not None and is_legal_transition(atom.status, PendingAtomStatus.FAILED):
            atom.status = PendingAtomStatus.FAILED

    async def on_cancelled(self, *, pending_alias: str, intent_id: str | None = None) -> None:
        """应用取消事件，保持既有 in-flight 状态机语义。"""
        atom = self._matching(pending_alias, intent_id)
        if atom is not None and is_legal_transition(atom.status, PendingAtomStatus.CANCELLED):
            atom.status = PendingAtomStatus.CANCELLED

    def _matching(self, alias: str, intent_id: str | None) -> PendingAtom | None:
        atom = self._atoms.get(alias)
        if atom is None and intent_id:
            atom = self._atoms.get(self._intent_aliases.get(intent_id, ""))
        if atom is not None and intent_id is not None and atom.intent_id != intent_id:
            logger.warning("写入意图事件的 intent_id 不匹配，已忽略: alias=%s", alias)
            return None
        return atom

    @property
    def size(self) -> int:
        """当前登记数量，包含保留的终态句柄。"""
        return len(self._atoms)


__all__ = ["WriteIntentRegistry"]
