from __future__ import annotations

import pytest

from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import (
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
    RuntimeEventBus,
)


def test_runtime_event_defaults_json_serializable():
    event = RuntimeEvent(
        event_type=RuntimeEventType.CHAT_RUN_CREATED,
        generation_id="gen-1",
    )

    data = event.model_dump(mode="json")

    assert event.event_id.startswith("evt_")
    assert data["event_type"] == "chat.run.created"
    assert data["generation_id"] == "gen-1"
    assert data["sequence"] == 0
    assert "timestamp" in data


def test_runtime_event_bus_assigns_sequence_and_replays_after_last_id():
    bus = RuntimeEventBus(buffer_size=10)
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))
    first, second = bus.buffer_snapshot

    assert first.sequence == 1
    assert second.sequence == 2

    sub = bus.subscribe(last_event_id=first.event_id)
    replay = sub._initial_events
    sub.close()

    assert [event.event_id for event in replay] == [second.event_id]


def test_runtime_event_bus_emits_gap_when_last_event_evicted():
    bus = RuntimeEventBus(buffer_size=1)
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    evicted_id = bus.buffer_snapshot[0].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))

    sub = bus.subscribe(last_event_id=evicted_id)
    replay = sub._initial_events
    sub.close()

    assert len(replay) == 1
    assert replay[0].event_type == RuntimeEventType.EVENT_STREAM_GAP
    assert replay[0].data["last_event_id"] == evicted_id
    assert replay[0].sequence == 3


@pytest.mark.asyncio
async def test_runtime_event_bus_slow_subscriber_drops_oldest():
    bus = RuntimeEventBus(buffer_size=10, subscriber_queue_size=1)
    sub = bus.subscribe(replay_last=0)

    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED, status="created"))
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))

    event = await sub._queue.get()
    sub.close()

    assert event.status == "streaming"
    assert sub.dropped_count == 1


def test_runtime_event_sinks_do_not_throw_and_record():
    null = NullRuntimeEventSink()
    null.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))

    recorder = RecordingRuntimeEventSink()
    recorder.scoped("system", component="test").emit(
        RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED)
    )

    assert recorder.events[0].subsystem == "system"
    assert recorder.events[0].component == "test"
