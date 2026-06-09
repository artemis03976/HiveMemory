"""Stable RuntimeEvent contract for run/task observability."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeEventType(str, Enum):
    CHAT_RUN_CREATED = "chat.run.created"
    CHAT_RUN_STATUS = "chat.run.status"
    CHAT_RUN_CANCEL_REQUESTED = "chat.run.cancel_requested"
    CHAT_RUN_CANCELLED = "chat.run.cancelled"
    CHAT_RUN_COMPLETED = "chat.run.completed"
    CHAT_RUN_FAILED = "chat.run.failed"

    AGENT_RUN_STARTED = "agent.run.started"
    AGENT_RUN_STATUS = "agent.run.status"
    AGENT_RUN_COMPLETED = "agent.run.completed"
    AGENT_RUN_CANCELLED = "agent.run.cancelled"
    AGENT_RUN_FAILED = "agent.run.failed"

    MEMORY_TASK_CREATED = "memory.task.created"
    MEMORY_TASK_STATUS = "memory.task.status"
    MEMORY_TASK_CANCEL_REQUESTED = "memory.task.cancel_requested"
    MEMORY_TASK_CANCELLED = "memory.task.cancelled"
    MEMORY_TASK_COMPLETED = "memory.task.completed"
    MEMORY_TASK_FAILED = "memory.task.failed"

    MEMORY_ATOM_SETTLED = "memory.atom.settled"
    MEMORY_ATOM_FAILED = "memory.atom.failed"
    MEMORY_ATOM_CANCELLED = "memory.atom.cancelled"

    EVENT_STREAM_GAP = "event.stream.gap"


def generate_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(default_factory=generate_event_id)
    sequence: int = 0
    event_type: RuntimeEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    span_name: str | None = None
    task_type: Literal["foreground", "background"] | None = None

    source: str | None = None
    subsystem: str | None = None
    component: str | None = None
    severity: Literal["debug", "info", "warning", "error"] = "info"

    generation_id: str | None = None
    agent_run_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    frame_id: str | None = None
    topic_id: str | None = None
    atom_id: str | None = None

    status: str | None = None
    reason: str | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "RuntimeEvent",
    "RuntimeEventType",
    "generate_event_id",
]
