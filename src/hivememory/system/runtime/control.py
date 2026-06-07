"""Runtime Control Registry — Phase 1 + Phase 2

Phase 1: Cancel Contract Hardening — ChatGenerationRun / RuntimeControlRegistry
Phase 2: MemoryGenerationJob runtime — MemoryGenerationJob / MemoryGenerationJobRegistry
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
# Phase 2: MemoryGenerationJob runtime
# ============================================================


class MemoryGenerationJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class MemoryTaskProgress:
    """单个 materialize task 的执行进度快照。"""

    pending_alias: str
    source_verb: str  # "WRITE" | "UPDATE"
    status: MemoryGenerationJobStatus = MemoryGenerationJobStatus.PENDING
    canonical_alias: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


@dataclass
class MemoryGenerationJob:
    """记忆后台生成任务的运行时句柄。"""

    job_id: str
    topic_id: str
    tasks: List[MemoryTaskProgress]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: MemoryGenerationJobStatus = MemoryGenerationJobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    _bg_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def request_cancel(self) -> None:
        if self.status not in (
            MemoryGenerationJobStatus.COMPLETED,
            MemoryGenerationJobStatus.FAILED,
            MemoryGenerationJobStatus.CANCELLED,
        ):
            self.cancel_event.set()

    def attach_task(self, task: asyncio.Task) -> None:
        self._bg_task = task


class MemoryGenerationJobRegistry:
    """进程内 memory generation job 注册表。"""

    def __init__(self, max_completed: int = 50) -> None:
        self._jobs: Dict[str, MemoryGenerationJob] = {}
        self._max_completed = max_completed

    def register(self, job: MemoryGenerationJob) -> None:
        self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[MemoryGenerationJob]:
        return self._jobs.get(job_id)

    def list_all(self) -> List[MemoryGenerationJob]:
        return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.request_cancel()
        return True

    def close(self, job_id: str, status: MemoryGenerationJobStatus) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.finished_at = datetime.now(timezone.utc)
        self._evict_old_completed()

    def _evict_old_completed(self) -> None:
        terminal = [
            j for j in self._jobs.values()
            if j.status in (
                MemoryGenerationJobStatus.COMPLETED,
                MemoryGenerationJobStatus.CANCELLED,
                MemoryGenerationJobStatus.FAILED,
            )
        ]
        if len(terminal) > self._max_completed:
            terminal.sort(key=lambda j: j.finished_at or j.created_at)
            for job in terminal[: len(terminal) - self._max_completed]:
                self._jobs.pop(job.job_id, None)


__all__ = [
    "ChatGenerationRun",
    "ChatGenerationRunStatus",
    "CancelResult",
    "RuntimeControlRegistry",
    "MemoryGenerationJobStatus",
    "MemoryTaskProgress",
    "MemoryGenerationJob",
    "MemoryGenerationJobRegistry",
]
