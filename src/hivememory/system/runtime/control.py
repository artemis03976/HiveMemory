"""Runtime control handles owned by the system application layer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
            if self.cancel_reason is None:
                self.cancel_reason = reason
        self.cancel_event.set()


class ChatGenerationRunRegistry:
    """In-process registry and cancellation surface for chat runs."""

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
            reason=run.cancel_reason or reason,
        )

    def close(self, generation_id: str, status: ChatGenerationRunStatus) -> None:
        run = self._runs.pop(generation_id, None)
        if run is not None and not run.cancelled:
            run.status = status


__all__ = [
    "CancelResult",
    "ChatGenerationRun",
    "ChatGenerationRunRegistry",
    "ChatGenerationRunStatus",
]
