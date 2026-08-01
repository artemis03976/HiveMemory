from __future__ import annotations

import asyncio
from collections.abc import Mapping
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


class QueueFrameEventSink:
    """Bounded FIFO sink used by Alice's streaming run driver."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any] | None],
        *,
        metadata: Mapping[str, Any] | None = None,
        sequence_start: int = 0,
    ) -> None:
        self._queue = queue
        self._metadata = dict(metadata or {})
        self._sequence = sequence_start

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
    "FrameEventSink",
    "NullFrameEventSink",
    "QueueFrameEventSink",
]
