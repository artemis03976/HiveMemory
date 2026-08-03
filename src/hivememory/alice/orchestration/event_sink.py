from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hivememory.agent_runtime.events import FrameEventSink


class ScopedFrameEventSink:
    """Project Alice-owned event metadata through an existing frame sink."""

    def __init__(
        self,
        sink: FrameEventSink,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        self._sink = sink
        self._metadata = dict(metadata)

    @property
    def wants_token_stream(self) -> bool:
        return self._sink.wants_token_stream

    async def emit(self, event: dict[str, Any]) -> None:
        await self._sink.emit(
            {
                **event,
                "data": {
                    **self._metadata,
                    **dict(event.get("data") or {}),
                },
            }
        )


__all__ = ["ScopedFrameEventSink"]
