from __future__ import annotations

import pytest

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.operations import RuntimeOperationObserver


def _observer(recorder: RecordingRuntimeEventSink | None = None) -> RuntimeOperationObserver:
    return RuntimeOperationObserver(
        recorder,
        subsystem="patchouli",
        component="patchouli_runtime",
        operation_key="patchouli.test_operation",
        operation_name="test_operation",
        operation_kind="test",
    )


@pytest.mark.asyncio
async def test_runtime_operation_observer_emits_started_and_completed():
    recorder = RecordingRuntimeEventSink()
    observer = _observer(recorder)

    async def run() -> dict[str, object]:
        return {"success": True, "value": object()}

    result = await observer.observe(
        run,
        started_data={"input_count": 1},
        summarize=lambda value: {
            "success": value["success"],
            "value": value["value"],
        },
    )

    assert result["success"] is True
    assert [event.event_type for event in recorder.events] == [
        RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
        RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
    ]
    started, completed = recorder.events
    assert started.source == "patchouli.test_operation"
    assert started.subsystem == "patchouli"
    assert started.component == "patchouli_runtime"
    assert started.data["operation_key"] == "patchouli.test_operation"
    assert started.data["operation_name"] == "test_operation"
    assert started.data["operation_kind"] == "test"
    assert started.data["input_count"] == 1
    assert completed.status == "completed"
    assert completed.data["success"] is True
    assert isinstance(completed.data["duration_ms"], float)
    assert isinstance(completed.data["value"], str)


@pytest.mark.asyncio
async def test_runtime_operation_observer_emits_failed_and_reraises():
    recorder = RecordingRuntimeEventSink()
    observer = _observer(recorder)

    async def run() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await observer.observe(
            run,
            failed_data=lambda exc: {"failure_stage": "run"},
        )

    assert [event.event_type for event in recorder.events] == [
        RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
        RuntimeEventType.SUBSYSTEM_OPERATION_FAILED,
    ]
    failed = recorder.events[-1]
    assert failed.status == "failed"
    assert failed.severity == "error"
    assert failed.reason == "boom"
    assert failed.data["success"] is False
    assert failed.data["failure_stage"] == "run"
    assert failed.data["error"] == "boom"


@pytest.mark.asyncio
async def test_runtime_operation_observer_accepts_dynamic_completion_metadata():
    recorder = RecordingRuntimeEventSink()
    observer = _observer(recorder)

    async def run() -> dict[str, int]:
        return {"timed_out": 1}

    await observer.observe(
        run,
        summarize=lambda value: {"timed_out": value["timed_out"]},
        completed_status=lambda value: (
            "completed_with_timeout"
            if value["timed_out"] > 0
            else "completed"
        ),
        completed_severity=lambda value: (
            "warning"
            if value["timed_out"] > 0
            else "info"
        ),
    )

    completed = recorder.events[-1]
    assert completed.status == "completed_with_timeout"
    assert completed.severity == "warning"
    assert completed.data["timed_out"] == 1


@pytest.mark.asyncio
async def test_runtime_operation_observer_default_null_sink_does_not_affect_run():
    observer = _observer()

    async def run() -> str:
        return "ok"

    assert await observer.observe(run) == "ok"
