"""Runtime Control Registry — Phase 1 + Phase 2

Phase 1: Cancel Contract Hardening — ChatGenerationRun / RuntimeControlRegistry
Phase 2: MemoryGenerationTask runtime — MemoryGenerationTask / MemoryGenerationTaskRegistry
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class ChatGenerationRunStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    STREAMING = "streaming"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class CancelResult:
    generation_id: str
    cancelled: bool
    status: str
    reason: str


@dataclass
class ChatGenerationRun:
    generation_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: ChatGenerationRunStatus = ChatGenerationRunStatus.CREATED
    cancel_reason: Optional[str] = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def request_cancel(self, reason: str = "user_requested") -> None:
        if self.status not in (
            ChatGenerationRunStatus.CANCELLED,
            ChatGenerationRunStatus.COMPLETED,
            ChatGenerationRunStatus.FAILED,
        ):
            self.status = ChatGenerationRunStatus.CANCELLING
            self.cancel_reason = reason
        self.cancel_event.set()


class RuntimeControlRegistry:
    """进程内 chat run 注册与取消控制面。"""

    def __init__(self) -> None:
        self._runs: dict[str, ChatGenerationRun] = {}

    def register(self, run: ChatGenerationRun) -> None:
        self._runs[run.generation_id] = run

    def get(self, generation_id: str) -> Optional[ChatGenerationRun]:
        return self._runs.get(generation_id)

    def cancel(self, generation_id: str, reason: str = "user_requested") -> CancelResult:
        run = self._runs.get(generation_id)
        if run is None:
            return CancelResult(
                generation_id=generation_id,
                cancelled=False,
                status="not_found",
                reason=reason,
            )
        run.request_cancel(reason)
        return CancelResult(
            generation_id=generation_id,
            cancelled=True,
            status=run.status.value,
            reason=reason,
        )

    def close(self, generation_id: str, status: ChatGenerationRunStatus) -> None:
        run = self._runs.pop(generation_id, None)
        if run is not None and not run.cancelled:
            run.status = status


# ============================================================
# Phase 2: MemoryGenerationTask runtime
# ============================================================


class MemoryGenerationTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MemoryGenerationSource(str, Enum):
    WRITE = "WRITE"      # MTP WRITE 主动链路
    UPDATE = "UPDATE"    # MTP UPDATE 主动链路
    ARCHIVE = "ARCHIVE"  # 话题被动归档 (Mode A)
    MERGE = "MERGE"      # 记忆合并（预留）
    SPLIT = "SPLIT"      # 记忆分裂（预留）


@dataclass
class MemoryTaskProgress:
    """单个生成子任务的执行进度快照。"""

    label: str                              # 可读标识：MTP 链路用 pending_alias，其余用 topic_id / memory_id
    source: MemoryGenerationSource
    pending_alias: Optional[str] = None    # 仅 WRITE / UPDATE 链路有值
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING
    canonical_alias: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


@dataclass
class MemoryGenerationTask:
    """记忆后台生成任务的运行时句柄。"""

    task_id: str
    topic_id: str
    tasks: List[MemoryTaskProgress]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    _bg_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def request_cancel(self) -> None:
        if self.status not in (
            MemoryGenerationTaskStatus.COMPLETED,
            MemoryGenerationTaskStatus.FAILED,
            MemoryGenerationTaskStatus.CANCELLED,
        ):
            self.cancel_event.set()

    def attach_task(self, task: asyncio.Task) -> None:
        self._bg_task = task


class MemoryGenerationTaskRegistry:
    """进程内 memory generation task 注册表。"""

    def __init__(self, max_completed: int = 50) -> None:
        self._tasks: Dict[str, MemoryGenerationTask] = {}
        self._max_completed = max_completed

    def register(self, memory_task: MemoryGenerationTask) -> None:
        self._tasks[memory_task.task_id] = memory_task

    def get(self, task_id: str) -> Optional[MemoryGenerationTask]:
        return self._tasks.get(task_id)

    def list_all(self) -> List[MemoryGenerationTask]:
        return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        memory_task = self._tasks.get(task_id)
        if memory_task is None:
            return False
        memory_task.request_cancel()
        return True

    def close(self, task_id: str, status: MemoryGenerationTaskStatus) -> None:
        memory_task = self._tasks.get(task_id)
        if memory_task is None:
            return
        memory_task.status = status
        memory_task.finished_at = datetime.now(timezone.utc)
        self._evict_old_completed()

    def _evict_old_completed(self) -> None:
        terminal = [
            j for j in self._tasks.values()
            if j.status in (
                MemoryGenerationTaskStatus.COMPLETED,
                MemoryGenerationTaskStatus.CANCELLED,
                MemoryGenerationTaskStatus.FAILED,
            )
        ]
        if len(terminal) > self._max_completed:
            terminal.sort(key=lambda j: j.finished_at or j.created_at)
            for memory_task in terminal[: len(terminal) - self._max_completed]:
                self._tasks.pop(memory_task.task_id, None)


__all__ = [
    "ChatGenerationRun",
    "ChatGenerationRunStatus",
    "CancelResult",
    "RuntimeControlRegistry",
    "MemoryGenerationTaskStatus",
    "MemoryGenerationSource",
    "MemoryTaskProgress",
    "MemoryGenerationTask",
    "MemoryGenerationTaskRegistry",
]
