from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CallRecordStatus(str, Enum):
    SUSPENDED = "suspended"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    APPLIED = "applied"
    CANCELLED = "cancelled"


@dataclass
class CallRecord:
    """Run-local state for one caller frame/action pair."""

    caller_frame_id: str
    action_id: str
    status: CallRecordStatus = CallRecordStatus.SUSPENDED

    @property
    def key(self) -> tuple[str, str]:
        return self.caller_frame_id, self.action_id

    def begin_resolution(self) -> None:
        if self.status != CallRecordStatus.SUSPENDED:
            raise RuntimeError(f"Cannot resolve CALL in state {self.status.value}.")
        self.status = CallRecordStatus.RESOLVING

    def mark_resolved(self) -> None:
        if self.status != CallRecordStatus.RESOLVING:
            raise RuntimeError(f"Cannot resolve CALL in state {self.status.value}.")
        self.status = CallRecordStatus.RESOLVED

    def mark_applied(self) -> None:
        if self.status != CallRecordStatus.RESOLVED:
            raise RuntimeError(f"Cannot apply CALL in state {self.status.value}.")
        self.status = CallRecordStatus.APPLIED

    def cancel(self) -> None:
        if self.status == CallRecordStatus.APPLIED:
            raise RuntimeError("An applied CALL cannot be cancelled.")
        self.status = CallRecordStatus.CANCELLED


__all__ = ["CallRecord", "CallRecordStatus"]
