from dataclasses import replace
from datetime import UTC, datetime

import pytest

from hivememory.patchouli.control.memory_generation.events import (
    MemoryTaskEventEmitter,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher


def _snapshot(
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING,
) -> MemoryGenerationTask:
    created_at = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
    return MemoryGenerationTask(
        task_id="task-1",
        topic_id="topic-1",
        label="memory-1",
        source=MemoryGenerationSource.WRITE,
        pending_alias="pending-1",
        status=status,
        canonical_alias=(
            "fact_memory_1"
            if status == MemoryGenerationTaskStatus.COMPLETED
            else None
        ),
        error=(
            "generation failed"
            if status == MemoryGenerationTaskStatus.FAILED
            else None
        ),
        created_at=created_at,
        started_at=(
            created_at
            if status != MemoryGenerationTaskStatus.PENDING
            else None
        ),
        finished_at=(
            created_at
            if status
            in {
                MemoryGenerationTaskStatus.COMPLETED,
                MemoryGenerationTaskStatus.CANCELLED,
                MemoryGenerationTaskStatus.FAILED,
            }
            else None
        ),
    )


def _emitter() -> tuple[MemoryTaskEventEmitter, RecordingRuntimeEventSink]:
    sink = RecordingRuntimeEventSink()
    return MemoryTaskEventEmitter(RuntimeEventPublisher(sink)), sink


def test_created_and_running_events_keep_stable_task_context() -> None:
    emitter, sink = _emitter()
    created = _snapshot()
    running = replace(created, status=MemoryGenerationTaskStatus.RUNNING)

    emitter.created(created)
    emitter.running(running)

    assert [event.event_type for event in sink.events] == [
        RuntimeEventType.MEMORY_TASK_CREATED,
        RuntimeEventType.MEMORY_TASK_STATUS,
    ]
    created_event, running_event = sink.events
    assert created_event.message == "Memory generation task created."
    assert running_event.message is None
    for event, snapshot in ((created_event, created), (running_event, running)):
        assert event.subsystem == "patchouli"
        assert event.source == "patchouli.memory_generation"
        assert event.component == "memory_task"
        assert event.task_type == "background"
        assert event.task_id == snapshot.task_id
        assert event.topic_id == snapshot.topic_id
        assert event.status == snapshot.status.value
        assert event.data["task_id"] == snapshot.task_id
        assert event.data["status"] == snapshot.status.value


@pytest.mark.parametrize(
    ("status", "event_type", "severity"),
    [
        (
            MemoryGenerationTaskStatus.COMPLETED,
            RuntimeEventType.MEMORY_TASK_COMPLETED,
            "info",
        ),
        (
            MemoryGenerationTaskStatus.CANCELLED,
            RuntimeEventType.MEMORY_TASK_CANCELLED,
            "info",
        ),
        (
            MemoryGenerationTaskStatus.FAILED,
            RuntimeEventType.MEMORY_TASK_FAILED,
            "error",
        ),
    ],
)
def test_terminal_event_maps_domain_status(
    status: MemoryGenerationTaskStatus,
    event_type: RuntimeEventType,
    severity: str,
) -> None:
    emitter, sink = _emitter()
    snapshot = _snapshot(status)

    emitter.terminal(snapshot, reason="terminal-reason")

    event = sink.events[0]
    assert event.event_type == event_type
    assert event.severity == severity
    assert event.reason == "terminal-reason"
    assert event.data["reason"] == "terminal-reason"


def test_cancel_requested_event_carries_reason_and_snapshot() -> None:
    emitter, sink = _emitter()
    snapshot = _snapshot().with_cancel_request("user_requested")

    emitter.cancel_requested(snapshot, reason="user_requested")

    event = sink.events[0]
    assert event.event_type == RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED
    assert event.reason == "user_requested"
    assert event.data["cancel_requested"] is True
    assert event.data["reason"] == "user_requested"


def test_terminal_rejects_non_terminal_snapshot() -> None:
    emitter, _ = _emitter()

    with pytest.raises(ValueError, match="not terminal"):
        emitter.terminal(_snapshot(MemoryGenerationTaskStatus.RUNNING))


def test_sink_failure_does_not_escape_emitter() -> None:
    class FailingSink:
        def emit(self, _event) -> None:
            raise RuntimeError("sink unavailable")

    emitter = MemoryTaskEventEmitter(RuntimeEventPublisher(FailingSink()))

    emitter.created(_snapshot())

