from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.server.routers.runtime_events import runtime_events_status, stream_runtime_events
from hivememory.system.config import HiveMemoryConfig, RuntimeEventsConfig
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import RuntimeEventBus


@pytest.mark.asyncio
async def test_runtime_events_status_enabled():
    bus = RuntimeEventBus()
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    result = await runtime_events_status(system=system)

    assert result["enabled"] is True
    assert result["latest_sequence"] == 1


@pytest.mark.asyncio
async def test_runtime_events_stream_disabled_returns_503():
    system = SimpleNamespace(
        runtime_events=None,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=False)),
    )

    response = await stream_runtime_events(
        request=SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        system=system,
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_runtime_events_stream_returns_sse_replay():
    bus = RuntimeEventBus()
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

    response = await stream_runtime_events(request=request, system=system)
    chunk = await response.body_iterator.__anext__()

    assert response.media_type == "text/event-stream"
    assert "event: runtime_event" in chunk
    assert "chat.run.created" in chunk
