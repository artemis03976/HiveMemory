"""Runtime Control Registry — Phase 1: Cancel Contract Hardening

单进程内存实现。管理 active chat runs 的 cancel token 与状态，
替换 ChatApplicationService._generation_events 的散装 dict 模式。
"""

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


__all__ = [
    "ChatGenerationRun",
    "ChatGenerationRunStatus",
    "CancelResult",
    "RuntimeControlRegistry",
]
