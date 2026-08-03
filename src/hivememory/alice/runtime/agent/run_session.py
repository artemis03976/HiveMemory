from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.alice.runtime.agent.call_record import CallRecord


class FrameSchedulingStatus(str, Enum):
    """Alice 编排层记录的单 frame 调度状态。"""

    PENDING = "pending"
    RUNNABLE = "runnable"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"


_ALLOWED_FRAME_TRANSITIONS: dict[
    FrameSchedulingStatus,
    frozenset[FrameSchedulingStatus],
] = {
    FrameSchedulingStatus.PENDING: frozenset(
        {FrameSchedulingStatus.RUNNABLE, FrameSchedulingStatus.TERMINATED}
    ),
    FrameSchedulingStatus.RUNNABLE: frozenset(
        {FrameSchedulingStatus.RUNNING, FrameSchedulingStatus.TERMINATED}
    ),
    FrameSchedulingStatus.RUNNING: frozenset(
        {
            FrameSchedulingStatus.WAITING,
            FrameSchedulingStatus.RUNNABLE,
            FrameSchedulingStatus.TERMINATED,
        }
    ),
    FrameSchedulingStatus.WAITING: frozenset(
        {FrameSchedulingStatus.RUNNABLE, FrameSchedulingStatus.TERMINATED}
    ),
    FrameSchedulingStatus.TERMINATED: frozenset(),
}


@dataclass
class RunSession:
    """Mutable state owned by exactly one Alice run."""

    agent_run_id: str
    generation_id: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    frames: dict[str, ExecutionFrame] = field(default_factory=dict)
    root_frame_id: str | None = None
    active_frame_id: str | None = None
    frame_statuses: dict[str, FrameSchedulingStatus] = field(default_factory=dict)
    call_records: dict[tuple[str, str], CallRecord] = field(default_factory=dict)
    call_records_by_callee: dict[str, CallRecord] = field(default_factory=dict)
    stream_sequence: int = 0

    def register_frame(self, frame: ExecutionFrame) -> None:
        """兼容登记普通 frame；拓扑明确时应使用 root/callee 专用入口。"""
        frame_id = frame.runtime_scope.frame_id
        self._validate_frame_for_registration(frame)
        existing = self.frames.get(frame_id)
        if existing is not None and existing is not frame:
            raise RuntimeError(f"Frame id already exists in this RunSession: {frame_id!r}")
        if existing is frame:
            return
        self.frames[frame_id] = frame
        self.frame_statuses[frame_id] = FrameSchedulingStatus.PENDING

    def register_root_frame(self, frame: ExecutionFrame) -> None:
        """登记本次 run 唯一的 root frame。"""
        if self.root_frame_id is not None:
            raise RuntimeError(f"RunSession already has a root frame: {self.root_frame_id!r}")
        self.register_frame(frame)
        self.root_frame_id = frame.runtime_scope.frame_id

    def register_callee_frame(self, frame: ExecutionFrame, record: CallRecord) -> None:
        """原子登记 callee，并建立 callee 到 CALL record 的反向索引。"""
        self._validate_frame_for_registration(frame)
        callee_frame_id = frame.runtime_scope.frame_id
        if callee_frame_id in self.frames:
            raise RuntimeError(
                f"Callee frame id already exists in this RunSession: {callee_frame_id!r}"
            )
        if self.call_records.get(record.key) is not record:
            raise ValueError(f"CALL record is not registered in this RunSession: {record.key!r}")
        if record.caller_frame_id not in self.frames:
            raise ValueError(
                f"Caller frame is not registered in this RunSession: {record.caller_frame_id!r}"
            )
        if callee_frame_id in self.call_records_by_callee:
            raise RuntimeError(f"Callee frame already has a CALL record: {callee_frame_id!r}")

        record.bind_callee(callee_frame_id)
        self.frames[callee_frame_id] = frame
        self.frame_statuses[callee_frame_id] = FrameSchedulingStatus.PENDING
        self.call_records_by_callee[callee_frame_id] = record

    def require_frame(self, frame: ExecutionFrame) -> None:
        frame_id = frame.runtime_scope.frame_id
        if self.frames.get(frame_id) is not frame:
            raise ValueError(
                f"Frame {frame_id!r} is not registered in RunSession {self.agent_run_id!r}."
            )

    def require_frame_status(
        self,
        frame_id: str,
        expected: FrameSchedulingStatus,
    ) -> None:
        actual = self.frame_statuses.get(frame_id)
        if actual != expected:
            raise RuntimeError(
                f"Frame {frame_id!r} has scheduling status {actual!r}, "
                f"expected {expected.value!r}."
            )

    def transition_frame(
        self,
        frame_id: str,
        target: FrameSchedulingStatus,
    ) -> None:
        """执行受保护的 run-local frame 状态转换。"""
        if frame_id not in self.frames:
            raise ValueError(f"Frame is not registered in this RunSession: {frame_id!r}")
        current = self.frame_statuses[frame_id]
        if target not in _ALLOWED_FRAME_TRANSITIONS[current]:
            raise RuntimeError(
                f"Cannot transition frame {frame_id!r} from "
                f"{current.value!r} to {target.value!r}."
            )
        if (
            target == FrameSchedulingStatus.RUNNING
            and self.active_frame_id is not None
            and self.active_frame_id != frame_id
        ):
            raise RuntimeError(
                "RunSession already has an active frame: " f"{self.active_frame_id!r}"
            )

        if current == FrameSchedulingStatus.RUNNING:
            self.active_frame_id = None
        if target == FrameSchedulingStatus.RUNNING:
            self.active_frame_id = frame_id
        self.frame_statuses[frame_id] = target

    def register_call(self, frame: ExecutionFrame, action_id: str) -> CallRecord:
        self.require_frame(frame)
        if not action_id:
            raise ValueError("A CALL action id cannot be empty.")
        key = (frame.runtime_scope.frame_id, action_id)
        if key in self.call_records:
            raise RuntimeError(f"CALL record already exists: {key!r}")
        record = CallRecord(caller_frame_id=key[0], action_id=key[1])
        self.call_records[key] = record
        return record

    def require_call(self, frame: ExecutionFrame, action_id: str) -> CallRecord:
        self.require_frame(frame)
        key = (frame.runtime_scope.frame_id, action_id)
        record = self.call_records.get(key)
        if record is None:
            raise ValueError(f"CALL record is not registered in this RunSession: {key!r}")
        return record

    def call_for_callee(self, callee_frame_id: str) -> CallRecord:
        record = self.call_records_by_callee.get(callee_frame_id)
        if record is None:
            raise ValueError(
                f"Callee frame has no CALL record in this RunSession: {callee_frame_id!r}"
            )
        return record

    def _validate_frame_for_registration(self, frame: ExecutionFrame) -> None:
        frame_id = frame.runtime_scope.frame_id
        if not frame_id:
            raise ValueError("A frame must have a frame_id before registration.")
        if frame.runtime_scope.run_id != self.agent_run_id:
            raise ValueError(
                "Frame run_id does not match its RunSession: "
                f"{frame.runtime_scope.run_id!r} != {self.agent_run_id!r}"
            )


__all__ = ["FrameSchedulingStatus", "RunSession"]
