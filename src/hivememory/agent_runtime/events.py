from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol


class FrameEventSink(Protocol):
    """Observation port for events produced while one frame is running."""

    @property
    def wants_token_stream(self) -> bool: ...

    async def emit(self, event: dict[str, Any]) -> None: ...


class NullFrameEventSink:
    @property
    def wants_token_stream(self) -> bool:
        return False

    async def emit(self, event: dict[str, Any]) -> None:
        del event


class CallbackFrameEventSink:
    """Compatibility sink backed by the existing async event callback."""

    def __init__(
        self,
        emitter: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        metadata: Mapping[str, Any] | None = None,
        wants_token_stream: bool = True,
    ) -> None:
        self._emitter = emitter
        self._metadata = dict(metadata or {})
        self._wants_token_stream = wants_token_stream

    @property
    def wants_token_stream(self) -> bool:
        return self._wants_token_stream

    async def emit(self, event: dict[str, Any]) -> None:
        if not self._metadata:
            await self._emitter(event)
            return
        projected = dict(event)
        projected["data"] = {
            **self._metadata,
            **dict(event.get("data") or {}),
        }
        await self._emitter(projected)


class QueueFrameEventSink:
    """Bounded FIFO sink used by Alice's streaming run driver."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any] | None],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._queue = queue
        self._metadata = dict(metadata or {})
        self._sequence = 0

    @property
    def wants_token_stream(self) -> bool:
        return True

    @property
    def next_sequence(self) -> int:
        return self._sequence

    async def emit(self, event: dict[str, Any]) -> None:
        data = {
            **self._metadata,
            **dict(event.get("data") or {}),
            "stream_sequence": self._sequence,
        }
        self._sequence += 1
        await self._queue.put({**event, "data": data})


__all__ = [
    "CallbackFrameEventSink",
    "FrameEventSink",
    "NullFrameEventSink",
    "QueueFrameEventSink",
]
