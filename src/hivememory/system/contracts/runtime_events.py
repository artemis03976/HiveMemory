"""Stable RuntimeEvent contract for run/task observability."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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

    COMMAND_EXECUTED = "command.executed"

    PASSIVE_INGRESS_EVENT_ACCEPTED = "passive.ingress.event.accepted"
    PASSIVE_INGRESS_DUPLICATE_IGNORED = "passive.ingress.duplicate.ignored"
    PASSIVE_MEMORY_CONTEXT_PREPARED = "passive.memory.context.prepared"
    PASSIVE_TURN_SUBMITTED = "passive.turn.submitted"
    PASSIVE_TURN_SUBMIT_FAILED = "passive.turn.submit_failed"

    GATEWAY_WORKFLOW_STARTED = "gateway.workflow.started"
    GATEWAY_STEP_COMPLETED = "gateway.step.completed"
    GATEWAY_WORKFLOW_COMPLETED = "gateway.workflow.completed"
    GATEWAY_WORKFLOW_CANCELLED = "gateway.workflow.cancelled"
    GATEWAY_WORKFLOW_FAILED = "gateway.workflow.failed"
    GATEWAY_ANALYSIS_CAPABILITY_COMPLETED = "gateway.analysis.capability.completed"

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

    WORK_QUEUED = "work.queued"
    WORK_STARTED = "work.started"
    WORK_RETRY_SCHEDULED = "work.retry_scheduled"
    WORK_SUCCEEDED = "work.succeeded"
    WORK_FAILED = "work.failed"
    WORK_DEAD_LETTERED = "work.dead_lettered"
    WORK_CANCEL_REQUESTED = "work.cancel_requested"
    WORK_CANCELLED = "work.cancelled"
    WORK_REJECTED = "work.rejected"

    MAINTENANCE_TASK_STARTED = "maintenance.task.started"
    MAINTENANCE_TASK_COMPLETED = "maintenance.task.completed"
    MAINTENANCE_TASK_FAILED = "maintenance.task.failed"

    SYSTEM_STARTING = "system.starting"
    SYSTEM_READY = "system.ready"
    SYSTEM_START_FAILED = "system.start_failed"
    SYSTEM_SHUTTING_DOWN = "system.shutting_down"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_STOP_FAILED = "system.stop_failed"

    SUBSYSTEM_OPERATION_STARTED = "subsystem.operation.started"
    SUBSYSTEM_OPERATION_COMPLETED = "subsystem.operation.completed"
    SUBSYSTEM_OPERATION_FAILED = "subsystem.operation.failed"

    EVENT_STREAM_GAP = "event.stream.gap"


def generate_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(default_factory=generate_event_id)
    sequence: int = 0
    event_type: RuntimeEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str | None = None
    span_name: str | None = None
    task_type: Literal["foreground", "background"] | None = None

    source: str | None = None
    subsystem: str | None = None
    component: str | None = None
    severity: Literal["debug", "info", "warning", "error"] = "info"

    generation_id: str | None = None
    interaction_id: str | None = None
    agent_run_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    frame_id: str | None = None
    topic_id: str | None = None
    atom_id: str | None = None
    # 观测投影：只标识资源归属域，不参与授权或业务幂等。
    workspace_id: str | None = None

    status: str | None = None
    reason: str | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "RuntimeEvent",
    "RuntimeEventType",
    "generate_event_id",
]
