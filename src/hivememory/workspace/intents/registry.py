"""进程级写入意图登记、派发与结算；只保存归属、发起者以及线程提交时的进程与线程关联。

登记在正常关闭时写盘、下次启动时读回：关闭时保存的生成工作在下次启动后继续执行并结算，
结算事件落到读回的句柄上。句柄登记满保留期后删除，之后读取由读取器按已过期交付。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from hivememory.components.bus.async_bus import AsyncSystemBus
from hivememory.config.files import rejected_path, write_atomically
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.core.models.pending import (
    PendingAtom,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
    UpdateFocus,
    WriteFocus,
    is_legal_transition,
)
from hivememory.utils.time import utc_now

logger = logging.getLogger(__name__)

#: 写入与更新句柄的 alias 前缀；登记中没有的这类 alias 由读取器按已过期交付。
WRITE_INTENT_ALIAS_PREFIXES = ("draft_", "rev_")

RECORD_FORMAT_VERSION = 1


def _slugify(text: str, max_len: int = 30) -> str:
    """把文本转为 alias 片段：小写，只保留字母、数字、空白与下划线，空白折叠为下划线。

    截取前 ``max_len`` 个字符并去掉首尾下划线；空片段统一回退为 untitled。
    """
    slug = re.sub(r"[^a-z0-9\s_]", "", text.lower().strip())
    slug = re.sub(r"\s+", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_")[:max_len].rstrip("_") or "untitled"


class WriteIntentRegistry:
    """全 Workspace 可回读的登记，同步操作在单个事件循环内原子完成。

    ``process_id`` 与 ``thread_id`` 仅供观测关联，不限制回读；不建进程的请求级提交
    没有进程与线程，二者为空。句柄自登记起保留 ``retention``，不论状态；登记新意图时
    删除满期的句柄。结算不回填 canonical 缓存。
    """

    def __init__(
        self,
        *,
        record_file: Path,
        retention: timedelta,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._record_file = record_file
        self._retention = retention
        self._now = now
        self._atoms: dict[str, PendingAtom] = {}
        self._intent_ids: set[str] = set()
        self._bus: AsyncSystemBus | None = None

    @property
    def record_file(self) -> Path:
        """登记写盘的文件；相对路径按工作目录（数据根）解析。"""
        return self._record_file

    def register_write(
        self,
        focus: WriteFocus,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None = None,
        thread_id: str | None = None,
    ) -> PendingAtom:
        """登记 WRITE 并进入 MATERIALIZING；返回独立副本，调用方不能改写登记中的状态。"""
        prefix = f"draft_{_slugify(focus.title or focus.content[:20])}"
        return self._register(prefix, focus, belong_to, from_actor, process_id, thread_id)

    def register_update(
        self,
        focus: UpdateFocus,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None = None,
        thread_id: str | None = None,
    ) -> PendingAtom:
        """登记 UPDATE；基础原子的可读性由能力层先行验证。"""
        return self._register(
            f"rev_{focus.base_alias}", focus, belong_to, from_actor, process_id, thread_id
        )

    def _register(
        self,
        prefix: str,
        focus: WriteFocus | UpdateFocus,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        process_id: str | None,
        thread_id: str | None,
    ) -> PendingAtom:
        self._drop_expired()
        # 保留期内 alias 不复用；短后缀冲突时重新生成，不能覆盖已结算的旧句柄。
        alias = f"{prefix}_{uuid4().hex[:4]}"
        while alias in self._atoms:
            alias = f"{prefix}_{uuid4().hex[:4]}"
        intent_id = f"intent_{uuid4().hex[:12]}"
        while intent_id in self._intent_ids:
            intent_id = f"intent_{uuid4().hex[:12]}"
        atom = PendingAtom(
            pending_alias=alias,
            intent_id=intent_id,
            status=PendingAtomStatus.MATERIALIZING,
            source_verb="WRITE" if isinstance(focus, WriteFocus) else "UPDATE",
            focus=focus.model_copy(deep=True),
            belong_to=belong_to,
            from_actor=from_actor,
            process_id=process_id,
            thread_id=thread_id,
            created_at=self._now(),
        )
        self._atoms[alias] = atom
        self._intent_ids.add(intent_id)
        return atom.model_copy(deep=True)

    def get(self, alias: str, workspace_identity: WorkspaceIdentity) -> PendingAtom | None:
        """只交付目标 Workspace 的记录；发起者、进程与线程均不参与授权。"""
        atom = self._atoms.get(alias)
        if atom is None or atom.belong_to != workspace_identity:
            return None
        return atom.model_copy(deep=True)

    def mark_failed(self, pending_alias: str, *, intent_id: str) -> None:
        """同步记录派发时的确定性拒绝，不带原因码；重复事件不能覆盖已经结算的终态。"""
        atom = self._matching(pending_alias, intent_id)
        if atom is not None and is_legal_transition(atom.status, PendingAtomStatus.FAILED):
            atom.status = PendingAtomStatus.FAILED

    def load(self) -> None:
        """读回上次正常关闭时写盘的登记并删除文件，必须在订阅结算事件之前调用。

        读回后立即删除文件：进程崩溃时这段时间的句柄随之丢失，与关闭时保存的生成工作
        一同丢失，下次启动不会读回一份过时的登记。满保留期的句柄不读回。文件无法解析
        或格式版本不认识时改名保留，登记以空表开始。
        """
        path = self.record_file
        if not path.exists():
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("write intent record must be an object")
            if document.get("format_version") != RECORD_FORMAT_VERSION:
                raise ValueError(
                    f"unknown write intent record format: {document.get('format_version')!r}"
                )
            items = document.get("intents")
            if not isinstance(items, list):
                raise ValueError("write intent record has no intents list")
            atoms = [PendingAtom.model_validate(item) for item in items]
        except (OSError, UnicodeDecodeError, ValueError):
            rejected = rejected_path(path)
            path.replace(rejected)
            logger.error(
                "读回写入句柄登记失败: reason=write_intents_unreadable, rejected_file=%s",
                rejected,
                exc_info=True,
            )
            return
        path.unlink()
        cutoff = self._cutoff()
        for atom in atoms:
            if atom.created_at > cutoff:
                self._atoms[atom.pending_alias] = atom
                self._intent_ids.add(atom.intent_id)
        logger.info("已读回写入句柄登记: read=%d, kept=%d", len(atoms), len(self._atoms))

    def save(self) -> None:
        """把保留期内的登记写盘，供下次启动读回；在解除结算订阅之后调用。

        登记为空时不写文件。写盘失败只记日志：关闭照常继续，这些句柄在下次启动后
        按已过期交付。
        """
        self._drop_expired()
        if not self._atoms:
            return
        document = {
            "format_version": RECORD_FORMAT_VERSION,
            "intents": [atom.model_dump(mode="json") for atom in self._atoms.values()],
        }
        try:
            write_atomically(
                self.record_file, json.dumps(document, ensure_ascii=False).encode("utf-8")
            )
        except OSError:
            logger.error(
                "写入句柄登记写盘失败: reason=write_intents_write_failed, path=%s, count=%d",
                self.record_file,
                len(self._atoms),
                exc_info=True,
            )

    def _cutoff(self) -> datetime:
        """登记时间不晚于这一时刻的句柄已满保留期。"""
        return self._now() - self._retention

    def _drop_expired(self) -> None:
        """删除满保留期的句柄，不论状态：在途过久的句柄也删除，不再伪装成仍在生成。"""
        cutoff = self._cutoff()
        expired = [alias for alias, atom in self._atoms.items() if atom.created_at <= cutoff]
        for alias in expired:
            self._intent_ids.discard(self._atoms.pop(alias).intent_id)

    def subscribe(self, bus: AsyncSystemBus) -> None:
        """订阅既有结算事件；重复装配幂等，切换总线时先取消旧订阅。"""
        if self._bus is bus:
            return
        self.unsubscribe()
        self._bus = bus
        bus.subscribe(GlobalEvents.PENDING_ATOM_SETTLED, self.on_settled)
        bus.subscribe(GlobalEvents.PENDING_ATOM_MERGED, self.on_merged)
        bus.subscribe(GlobalEvents.PENDING_ATOM_FAILED, self.on_failed)
        bus.subscribe(GlobalEvents.PENDING_ATOM_CANCELLED, self.on_cancelled)

    def unsubscribe(self) -> None:
        """解除订阅，不删除已经登记或结算的句柄。"""
        if self._bus is None:
            return
        self._bus.unsubscribe(GlobalEvents.PENDING_ATOM_SETTLED, self.on_settled)
        self._bus.unsubscribe(GlobalEvents.PENDING_ATOM_MERGED, self.on_merged)
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

    async def on_failed(
        self, *, pending_alias: str, intent_id: str, reason: str | None = None
    ) -> None:
        """严格匹配 intent_id 后应用失败事件并记下原因码；重复事件不能覆盖已经结算的终态。"""
        atom = self._matching(pending_alias, intent_id)
        if atom is not None and is_legal_transition(atom.status, PendingAtomStatus.FAILED):
            atom.status = PendingAtomStatus.FAILED
            atom.failure_reason = reason

    async def on_merged(
        self, *, settlement: PendingAtomSettlement, absorbed_memory_uuid: str
    ) -> None:
        """只替换仍指向被吸收记忆的 CREATED 结算；改指不构成状态迁移。"""
        atom = self._matching(settlement.pending_alias, settlement.intent_id)
        if (
            atom is None
            or atom.status != PendingAtomStatus.SETTLED
            or atom.settlement is None
            or atom.settlement.resolution != PendingAtomResolution.CREATED
            or atom.settlement.canonical_uuid != absorbed_memory_uuid
        ):
            return
        atom.settlement = settlement.model_copy(deep=True)

    async def on_cancelled(self, *, pending_alias: str, intent_id: str) -> None:
        """严格匹配 intent_id 后应用取消事件，保持既有 in-flight 状态机语义。"""
        atom = self._matching(pending_alias, intent_id)
        if atom is not None and is_legal_transition(atom.status, PendingAtomStatus.CANCELLED):
            atom.status = PendingAtomStatus.CANCELLED

    def _matching(self, alias: str, intent_id: str) -> PendingAtom | None:
        """alias 在保留期内不复用，intent_id 再区分过时事件；两者都匹配才返回登记。"""
        atom = self._atoms.get(alias)
        if atom is not None and atom.intent_id != intent_id:
            logger.warning("写入意图事件的 intent_id 不匹配，已忽略: alias=%s", alias)
            return None
        return atom

    @property
    def size(self) -> int:
        """当前登记数量，包含保留期内的终态句柄。"""
        return len(self._atoms)


__all__ = ["WRITE_INTENT_ALIAS_PREFIXES", "WriteIntentRegistry"]
