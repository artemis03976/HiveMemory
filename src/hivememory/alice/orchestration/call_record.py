from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CallRecordStatus(str, Enum):
    """CALL 记账生命周期：挂起 → 解析中 → 已结算 → 已回填 / 已取消。"""

    SUSPENDED = "suspended"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    APPLIED = "applied"
    CANCELLED = "cancelled"


@dataclass
class CallRecord:
    """一个 caller frame/action 对在 run 内的 CALL 记账状态。"""

    caller_frame_id: str
    action_id: str
    status: CallRecordStatus = CallRecordStatus.SUSPENDED
    callee_frame_id: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.caller_frame_id, self.action_id

    def begin_resolution(self) -> None:
        if self.status != CallRecordStatus.SUSPENDED:
            raise RuntimeError(f"Cannot resolve CALL in state {self.status.value}.")
        self.status = CallRecordStatus.RESOLVING

    def bind_callee(self, callee_frame_id: str) -> None:
        """在解析阶段把 CALL 与唯一的 callee frame 绑定。"""
        if self.status != CallRecordStatus.RESOLVING:
            raise RuntimeError(f"Cannot bind callee in CALL state {self.status.value}.")
        if not callee_frame_id:
            raise ValueError("A callee frame id cannot be empty.")
        if callee_frame_id == self.caller_frame_id:
            raise ValueError("Caller and callee frame ids must be different.")
        if self.callee_frame_id is not None:
            raise RuntimeError(
                "CALL record already has a callee frame: " f"{self.callee_frame_id!r}"
            )
        self.callee_frame_id = callee_frame_id

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
        if self.status == CallRecordStatus.CANCELLED:
            return
        self.status = CallRecordStatus.CANCELLED


__all__ = ["CallRecord", "CallRecordStatus"]
