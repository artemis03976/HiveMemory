"""Best-effort RuntimeEvent bus and sinks."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable, Protocol

from hivememory.infrastructure.trace_context import (
    current_span_name,
    current_task_type,
    current_trace_id,
)
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType

logger = logging.getLogger(__name__)


def safe_runtime_event_value(value: Any) -> Any:
    # RuntimeEvent payload 只保留 JSON 友好的摘要值，复杂对象降级为 repr。
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): safe_runtime_event_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            safe_runtime_event_value(item)
            for item in value
        ]
    return repr(value)


class RuntimeEventSink(Protocol):
    def emit(self, event: RuntimeEvent) -> None:
        """Emit an observational event without affecting business flow."""

    def scoped(
        self,
        subsystem: str | None = None,
        *,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventSink:
        """Return a sink that fills source metadata."""


class NullRuntimeEventSink:
    """RuntimeEvent sink used when the transport is disabled."""

    def emit(self, event: RuntimeEvent) -> None:
        return

    def scoped(
        self,
        subsystem: str | None = None,
        *,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventSink:
        return self


class RecordingRuntimeEventSink:
    """Test helper sink that records emitted events."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def scoped(
        self,
        subsystem: str | None = None,
        *,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventSink:
        return ScopedRuntimeEventSink(
            self,
            subsystem=subsystem,
            source=source,
            component=component,
        )


class RuntimeEventSubscription:
    def __init__(
        self,
        bus: RuntimeEventBus,
        queue: asyncio.Queue[RuntimeEvent],
        initial_events: Iterable[RuntimeEvent],
    ) -> None:
        self._bus = bus
        self._queue = queue
        self._initial_events = list(initial_events)
        self.dropped_count = 0
        self.closed = False

    def drop_oldest_and_put(self, event: RuntimeEvent) -> None:
        if self.closed:
            return
        try:
            self._queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        with suppress(asyncio.QueueEmpty):
            self._queue.get_nowait()
            self.dropped_count += 1
        with suppress(asyncio.QueueFull):
            self._queue.put_nowait(event)

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        try:
            for event in self._initial_events:
                yield event
            while True:
                yield await self._queue.get()
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._bus.unsubscribe(self)


class RuntimeEventBus:
    """Independent best-effort bus for runtime observability events."""

    def __init__(
        self,
        *,
        buffer_size: int = 1000,
        subscriber_queue_size: int = 100,
    ) -> None:
        self._buffer: deque[RuntimeEvent] = deque(maxlen=max(1, buffer_size))
        self._subscriber_queue_size = max(1, subscriber_queue_size)
        self._subscriptions: set[RuntimeEventSubscription] = set()
        self._sequence = 0

    @property
    def latest_sequence(self) -> int:
        return self._sequence

    @property
    def buffer_snapshot(self) -> list[RuntimeEvent]:
        return list(self._buffer)

    def emit(self, event: RuntimeEvent) -> None:
        try:
            prepared = self._prepare_event(event)
            self._buffer.append(prepared)
            for subscription in list(self._subscriptions):
                subscription.drop_oldest_and_put(prepared)
        except Exception:
            logger.warning("RuntimeEventBus emit failed", exc_info=True)

    def scoped(
        self,
        subsystem: str | None = None,
        *,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventSink:
        return ScopedRuntimeEventSink(
            self,
            subsystem=subsystem,
            source=source,
            component=component,
        )

    def subscribe(
        self,
        *,
        replay_last: int = 100,
        last_event_id: str | None = None,
    ) -> RuntimeEventSubscription:
        initial_events = self._replay_events(
            replay_last=replay_last,
            last_event_id=last_event_id,
        )
        subscription = RuntimeEventSubscription(
            self,
            asyncio.Queue(maxsize=self._subscriber_queue_size),
            initial_events,
        )
        self._subscriptions.add(subscription)
        return subscription

    async def stream(
        self,
        *,
        replay_last: int = 100,
        last_event_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        subscription = self.subscribe(
            replay_last=replay_last,
            last_event_id=last_event_id,
        )
        async for event in subscription.events():
            yield event

    def unsubscribe(self, subscription: RuntimeEventSubscription) -> None:
        self._subscriptions.discard(subscription)

    def _prepare_event(self, event: RuntimeEvent) -> RuntimeEvent:
        self._sequence += 1
        update = {
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc),
        }
        if not event.trace_id:
            update["trace_id"] = current_trace_id.get()
        if not event.span_name:
            update["span_name"] = current_span_name.get()
        if not event.task_type:
            update["task_type"] = current_task_type.get()
        return event.model_copy(update=update)

    def _replay_events(
        self,
        *,
        replay_last: int,
        last_event_id: str | None,
    ) -> list[RuntimeEvent]:
        buffer = list(self._buffer)
        if not buffer:
            return []

        if last_event_id:
            for index, event in enumerate(buffer):
                if event.event_id == last_event_id:
                    return buffer[index + 1 :]
            return [self._build_gap_event(last_event_id, buffer)]

        replay_count = max(0, replay_last)
        if replay_count == 0:
            return []
        return buffer[-replay_count:]

    def _build_gap_event(
        self,
        last_event_id: str,
        buffer: list[RuntimeEvent],
    ) -> RuntimeEvent:
        earliest = buffer[0]
        dropped_count = max(0, self._sequence - len(buffer))
        return self._prepare_event(
            RuntimeEvent(
                event_type=RuntimeEventType.EVENT_STREAM_GAP,
                severity="warning",
                message="Runtime event replay gap detected.",
                data={
                    "last_event_id": last_event_id,
                    "earliest_event_id": earliest.event_id,
                    "dropped_count": dropped_count,
                },
            )
        )


class ScopedRuntimeEventSink:
    def __init__(
        self,
        sink: RuntimeEventSink,
        *,
        subsystem: str | None = None,
        source: str | None = None,
        component: str | None = None,
    ) -> None:
        self._sink = sink
        self._subsystem = subsystem
        self._source = source
        self._component = component

    def emit(self, event: RuntimeEvent) -> None:
        event = event.model_copy(
            update={
                "subsystem": event.subsystem or self._subsystem,
                "source": event.source or self._source or self._subsystem,
                "component": event.component or self._component,
            }
        )
        self._sink.emit(event)

    def scoped(
        self,
        subsystem: str | None = None,
        *,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventSink:
        return ScopedRuntimeEventSink(
            self._sink,
            subsystem=subsystem or self._subsystem,
            source=source or self._source,
            component=component or self._component,
        )


__all__ = [
    "NullRuntimeEventSink",
    "RecordingRuntimeEventSink",
    "RuntimeEventBus",
    "RuntimeEventSink",
    "RuntimeEventSubscription",
    "ScopedRuntimeEventSink",
    "safe_runtime_event_value",
]
