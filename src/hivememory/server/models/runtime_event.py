"""RuntimeEvent API models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from hivememory.system.contracts.runtime_events import RuntimeEvent


class RuntimeEventResponse(BaseModel):
    event_id: str
    sequence: int
    event_type: str
    timestamp: datetime
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

    @classmethod
    def from_domain(cls, event: RuntimeEvent) -> "RuntimeEventResponse":
        return cls.model_validate(event.model_dump(mode="json"))


class RuntimeEventStatusResponse(BaseModel):
    enabled: bool
    buffer_size: int
    subscriber_queue_size: int
    latest_sequence: int | None = None


class RuntimeEventDisabledResponse(BaseModel):
    status: Literal["disabled"] = "disabled"
    detail: str


__all__ = [
    "RuntimeEventDisabledResponse",
    "RuntimeEventResponse",
    "RuntimeEventStatusResponse",
]
