from __future__ import annotations

from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher


def test_agent_run_event_emitter_binds_run_context_and_terminal_summary() -> None:
    sink = RecordingRuntimeEventSink()
    emitter = AgentRunEventEmitter(
        RuntimeEventPublisher(sink).scoped(
            subsystem="alice",
            component="agent_run_service",
        )
    )
    events = emitter.for_run(
        agent_run_id="run-1",
        generation_id="generation-1",
        topic_id="topic-1",
        agent_id="agent-1",
        workspace_id="isolation_workspace",
    )
    result = AgentRunResult(
        status=AgentRunStatus.COMPLETED,
        mtp_iterations=2,
        total_iterations=3,
    )

    events.started()
    events.completed(result)

    assert [event.event_type for event in sink.events] == [
        RuntimeEventType.AGENT_RUN_STARTED,
        RuntimeEventType.AGENT_RUN_COMPLETED,
    ]
    assert all(event.subsystem == "alice" for event in sink.events)
    assert all(event.component == "agent_run_service" for event in sink.events)
    assert all(event.agent_run_id == "run-1" for event in sink.events)
    assert all(event.generation_id == "generation-1" for event in sink.events)
    assert all(event.interaction_id == "generation-1" for event in sink.events)
    assert all(event.workspace_id == "isolation_workspace" for event in sink.events)
    assert all(event.task_type == "foreground" for event in sink.events)
    assert sink.events[-1].data == {
        "mtp_iterations": 2,
        "total_iterations": 3,
        "materialize_task_count": 0,
    }


def test_agent_run_event_emitter_records_stream_close_without_business_effects() -> None:
    sink = RecordingRuntimeEventSink()
    events = AgentRunEventEmitter(RuntimeEventPublisher(sink)).for_run(
        agent_run_id="run-1",
        generation_id=None,
        topic_id=None,
        agent_id=None,
    )

    events.cancelled(message="closed", close_reason="stream_closed")

    event = sink.events[0]
    assert event.event_type == RuntimeEventType.AGENT_RUN_CANCELLED
    assert event.status == "cancelled"
    assert event.message == "closed"
    assert event.data == {"close_reason": "stream_closed"}
