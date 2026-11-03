from __future__ import annotations

import json
import logging

import pytest

from hivememory.components.events.bus import (
    RUNTIME_EVENT_RECORD_LOGGER,
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
    RuntimeEventBus,
    safe_runtime_event_value,
)
from hivememory.core.contracts.runtime_events import (
    RuntimeEvent,
    RuntimeEventType,
)


def test_runtime_event_defaults_json_serializable():
    event = RuntimeEvent(
        event_type=RuntimeEventType.CHAT_RUN_CREATED,
        process_id="process-1",
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


class _MessageHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_runtime_event_bus_records_each_stamped_event_as_one_json_line():
    """记录的是总线补全序号、时间戳与 scope 之后的事件，与缓冲中的同一份一致。"""
    handler = _MessageHandler()
    record_logger = logging.getLogger(RUNTIME_EVENT_RECORD_LOGGER)
    record_logger.addHandler(handler)
    try:
        bus = RuntimeEventBus(buffer_size=10)
        bus.scoped("patchouli", component="maintenance_scheduler").emit(
            RuntimeEvent(event_type=RuntimeEventType.MAINTENANCE_TASK_COMPLETED, task_id="t-1")
        )
    finally:
        record_logger.removeHandler(handler)

    (line,) = handler.messages
    (buffered,) = bus.buffer_snapshot
    assert json.loads(line) == buffered.model_dump(mode="json")
    assert buffered.sequence == 1
    assert buffered.subsystem == "patchouli"


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
