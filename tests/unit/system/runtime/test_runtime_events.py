from __future__ import annotations

import pytest

from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import (
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
    RuntimeEventBus,
    safe_runtime_event_value,
)


def test_runtime_event_defaults_json_serializable():
    event = RuntimeEvent(
        event_type=RuntimeEventType.CHAT_RUN_CREATED,
        generation_id="gen-1",
    )

    assert event.event_id.startswith("evt_")
    # model_dump(mode="json") 序列化契约：timestamp 渲染为字符串
    data = event.model_dump(mode="json")
    assert isinstance(data["timestamp"], str)


@pytest.mark.asyncio
async def test_runtime_event_bus_assigns_sequence_and_replays_after_last_id():
    bus = RuntimeEventBus(buffer_size=10)
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))
    first, second = bus.buffer_snapshot

    assert first.sequence == 1
    assert second.sequence == 2

    sub = bus.subscribe(last_event_id=first.event_id)
    stream = sub.events()
    replay_event = await stream.__anext__()
    await stream.aclose()

    assert replay_event.event_id == second.event_id


@pytest.mark.asyncio
async def test_runtime_event_bus_emits_gap_when_last_event_evicted():
    bus = RuntimeEventBus(buffer_size=1)
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    evicted_id = bus.buffer_snapshot[0].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))

    sub = bus.subscribe(last_event_id=evicted_id)
    stream = sub.events()
    replay_event = await stream.__anext__()
    await stream.aclose()

    assert replay_event.event_type == RuntimeEventType.EVENT_STREAM_GAP
    assert replay_event.data["last_event_id"] == evicted_id
    assert replay_event.sequence == 3


@pytest.mark.asyncio
async def test_runtime_event_bus_slow_subscriber_drops_oldest():
    bus = RuntimeEventBus(buffer_size=10, subscriber_queue_size=1)
    sub = bus.subscribe(replay_last=0)

    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED, status="created"))
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))

    stream = sub.events()
    event = await stream.__anext__()
    await stream.aclose()

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


def test_safe_runtime_event_value_normalizes_nested_payload():
    class CustomValue:
        def __repr__(self) -> str:
            return "<custom>"

    value = safe_runtime_event_value(
        {
            1: ("ok", CustomValue()),
            "nested": [{"flag": True}],
        }
    )

    assert value == {
        "1": ["ok", "<custom>"],
        "nested": [{"flag": True}],
    }
