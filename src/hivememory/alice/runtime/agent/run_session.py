from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.alice.runtime.agent.call_record import CallRecord


@dataclass
class RunSession:
    """Mutable state owned by exactly one Alice run."""

    agent_run_id: str
    generation_id: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    frames: dict[str, ExecutionFrame] = field(default_factory=dict)
    call_records: dict[tuple[str, str], CallRecord] = field(default_factory=dict)
    stream_sequence: int = 0

    def register_frame(self, frame: ExecutionFrame) -> None:
        frame_id = frame.runtime_scope.frame_id
        if not frame_id:
            raise ValueError("A frame must have a frame_id before registration.")
        if frame.runtime_scope.run_id != self.agent_run_id:
            raise ValueError(
                "Frame run_id does not match its RunSession: "
                f"{frame.runtime_scope.run_id!r} != {self.agent_run_id!r}"
            )
        self.frames[frame_id] = frame

    def register_call(self, frame: ExecutionFrame, action_id: str) -> CallRecord:
        key = (frame.runtime_scope.frame_id, action_id)
        if key in self.call_records:
            raise RuntimeError(f"CALL record already exists: {key!r}")
        record = CallRecord(caller_frame_id=key[0], action_id=key[1])
        self.call_records[key] = record
        return record


__all__ = ["RunSession"]
