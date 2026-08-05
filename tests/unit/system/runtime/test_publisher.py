from __future__ import annotations

from pydantic import BaseModel

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher


class _Payload(BaseModel):
    count: int


def test_runtime_event_publisher_combines_scope_context_and_typed_payload() -> None:
    sink = RecordingRuntimeEventSink()
    publisher = (
        RuntimeEventPublisher(sink)
        .scoped(subsystem="alice", component="agent_run")
        .bind(
            task_type="foreground",
            generation_id="generation-1",
            agent_run_id="run-1",
        )
    )

    publisher.emit(
        RuntimeEventType.AGENT_RUN_STARTED,
        status="started",
        data=_Payload(count=2),
    )

    event = sink.events[0]
    assert event.subsystem == "alice"
    assert event.source == "alice"
    assert event.component == "agent_run"
    assert event.task_type == "foreground"
    assert event.generation_id == "generation-1"
    assert event.agent_run_id == "run-1"
    assert event.data == {"count": 2}


def test_runtime_event_publisher_sanitizes_mapping_payload() -> None:
    class CustomValue:
        def __repr__(self) -> str:
            return "<custom>"

    sink = RecordingRuntimeEventSink()

    RuntimeEventPublisher(sink).emit(
        RuntimeEventType.AGENT_RUN_FAILED,
        data={"error": CustomValue(), "nested": (1, 2)},
    )

    assert sink.events[0].data == {"error": "<custom>", "nested": [1, 2]}


def test_runtime_event_publisher_isolates_sink_failure() -> None:
    class FailingSink:
        def emit(self, _event) -> None:
            raise RuntimeError("sink unavailable")

        def scoped(self, *_args, **_kwargs):
            return self

    RuntimeEventPublisher(FailingSink()).emit(RuntimeEventType.AGENT_RUN_STARTED)
