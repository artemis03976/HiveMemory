"""Patchouli-owned runtime handles for memory generation tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class MemoryGenerationTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MemoryGenerationSource(str, Enum):
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    ARCHIVE = "ARCHIVE"
    MERGE = "MERGE"
    SPLIT = "SPLIT"


@dataclass
class MemoryGenerationTask:
    """Runtime handle for one Patchouli memory generation task."""

    task_id: str
    topic_id: str
    label: str
    source: MemoryGenerationSource
    pending_alias: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING
    canonical_alias: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    _bg_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)
    _terminal_status_published: bool = field(default=False, repr=False, compare=False)

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

    def cancel_background_task(self) -> None:
        if self._bg_task is not None and not self._bg_task.done():
            self._bg_task.cancel()


class MemoryGenerationTaskDTO(BaseModel):
    """Stable public projection for MemoryGenerationTask API and RuntimeEvent payloads."""

    task_id: str
    topic_id: str
    label: str
    source: str
    pending_alias: Optional[str] = None
    status: str
    canonical_alias: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_requested: bool = False
    cancelled: bool = False
    reason: Optional[str] = None


def memory_task_to_dto(
    memory_task: MemoryGenerationTask,
    *,
    reason: str | None = None,
) -> MemoryGenerationTaskDTO:
    """Serialize one memory task into the shared frontend-facing DTO."""
    cancel_requested = memory_task.cancelled
    cancelled = memory_task.status == MemoryGenerationTaskStatus.CANCELLED
    return MemoryGenerationTaskDTO(
        task_id=memory_task.task_id,
        topic_id=memory_task.topic_id,
        label=memory_task.label,
        source=memory_task.source.value,
        pending_alias=memory_task.pending_alias,
        status=memory_task.status.value,
        canonical_alias=memory_task.canonical_alias,
        error=memory_task.error,
        created_at=memory_task.created_at.isoformat(),
        started_at=(
            memory_task.started_at.isoformat()
            if memory_task.started_at is not None
            else None
        ),
        finished_at=(
            memory_task.finished_at.isoformat()
            if memory_task.finished_at is not None
            else None
        ),
        cancel_requested=cancel_requested,
        cancelled=cancelled,
        reason=reason if reason is not None else (
            "user_requested" if cancel_requested else None
        ),
    )


class MemoryGenerationTaskRegistry:
    """In-process registry for Patchouli memory generation tasks."""

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
        memory_task.cancel_background_task()
        return True

    def close(self, task_id: str, status: MemoryGenerationTaskStatus) -> None:
        memory_task = self._tasks.get(task_id)
        if memory_task is None:
            return
        memory_task.status = status
        if memory_task.finished_at is None:
            memory_task.finished_at = datetime.now(timezone.utc)
        self._evict_old_completed()

    def _evict_old_completed(self) -> None:
        terminal = [
            task
            for task in self._tasks.values()
            if task.status
            in (
                MemoryGenerationTaskStatus.COMPLETED,
                MemoryGenerationTaskStatus.CANCELLED,
                MemoryGenerationTaskStatus.FAILED,
            )
        ]
        if len(terminal) <= self._max_completed:
            return
        terminal.sort(key=lambda task: task.finished_at or task.created_at)
        for memory_task in terminal[: len(terminal) - self._max_completed]:
            self._tasks.pop(memory_task.task_id, None)


__all__ = [
    "MemoryGenerationSource",
    "MemoryGenerationTask",
    "MemoryGenerationTaskDTO",
    "MemoryGenerationTaskRegistry",
    "MemoryGenerationTaskStatus",
    "memory_task_to_dto",
]
