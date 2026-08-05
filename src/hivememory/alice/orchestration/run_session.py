from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.alice.orchestration.sub_agent.call_record import CallRecord, CallRecordStatus


@dataclass
class RunSession:
    """一次 Alice run 独占的可变状态（run-local 控制面）。

    保存帧注册表、CALL 记账与取消信号。frame 的挂起与恢复由 run-local
    递归执行器的协程栈表达，Session 不充当调度程序计数器。
    """

    agent_run_id: str
    generation_id: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    frames: dict[str, ExecutionFrame] = field(default_factory=dict)
    root_frame_id: str | None = None
    call_records: dict[tuple[str, str], CallRecord] = field(default_factory=dict)
    call_records_by_callee: dict[str, CallRecord] = field(default_factory=dict)

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
        self.call_records_by_callee[callee_frame_id] = record

    def require_frame(self, frame: ExecutionFrame) -> None:
        frame_id = frame.runtime_scope.frame_id
        if self.frames.get(frame_id) is not frame:
            raise ValueError(
                f"Frame {frame_id!r} is not registered in RunSession {self.agent_run_id!r}."
            )

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

    def cancel_unapplied_calls(self) -> None:
        """协程取消时只终止当前 run 中尚未回填的 CALL。"""
        for record in self.call_records.values():
            if record.status not in {CallRecordStatus.APPLIED, CallRecordStatus.CANCELLED}:
                record.cancel()

    def _validate_frame_for_registration(self, frame: ExecutionFrame) -> None:
        frame_id = frame.runtime_scope.frame_id
        if not frame_id:
            raise ValueError("A frame must have a frame_id before registration.")
        if frame.runtime_scope.run_id != self.agent_run_id:
            raise ValueError(
                "Frame run_id does not match its RunSession: "
                f"{frame.runtime_scope.run_id!r} != {self.agent_run_id!r}"
            )


__all__ = ["RunSession"]
