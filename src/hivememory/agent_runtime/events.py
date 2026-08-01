from __future__ import annotations

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
            **dict(event.get("data") or {}),
            **self._metadata,
        }
        await self._emitter(projected)


__all__ = [
    "CallbackFrameEventSink",
    "FrameEventSink",
    "NullFrameEventSink",
]
