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
    # scoped 链路：subsystem/component 填充；source 回退到 subsystem
    assert event.subsystem == "alice"
    assert event.source == "alice"
    assert event.component == "agent_run"


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
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, _event) -> None:
            self.calls += 1
            raise RuntimeError("sink unavailable")

        def scoped(self, *_args, **_kwargs):
            return self

    sink = FailingSink()
    RuntimeEventPublisher(sink).emit(RuntimeEventType.AGENT_RUN_STARTED)

    # 异常被吞掉，且事件确实到达了 sink
    assert sink.calls == 1
