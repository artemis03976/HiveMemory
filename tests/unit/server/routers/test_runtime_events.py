from __future__ import annotations

import json
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

    assert result.enabled is True
    assert result.latest_sequence == 1


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

    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_runtime_events_stream_preserves_workspace_correlations():
    """确认 SSE 重放保留观测所需的 Workspace 与 Interaction 投影。"""
    bus = RuntimeEventBus()
    bus.emit(
        RuntimeEvent(
            event_type=RuntimeEventType.CHAT_RUN_CREATED,
            interaction_id="interaction-isolated",
            workspace_id="isolation_workspace",
        )
    )
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

    response = await stream_runtime_events(request=request, system=system)
    chunk = await response.body_iterator.__anext__()
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    payload = json.loads(data_line.removeprefix("data: "))

    assert payload["interaction_id"] == "interaction-isolated"
    assert payload["workspace_id"] == "isolation_workspace"

    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_runtime_events_stream_replays_after_last_event_id_query():
    bus = RuntimeEventBus()
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    first_id = bus.buffer_snapshot[0].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

    response = await stream_runtime_events(
        request=request,
        last_event_id_query=first_id,
        system=system,
    )
    chunk = await response.body_iterator.__anext__()

    assert "chat.run.status" in chunk
    assert "streaming" in chunk
    assert first_id not in chunk

    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_runtime_events_stream_header_last_event_id_emits_gap_when_evicted():
    bus = RuntimeEventBus(buffer_size=1)
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    evicted_id = bus.buffer_snapshot[0].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

    response = await stream_runtime_events(
        request=request,
        last_event_id_header=evicted_id,
        system=system,
    )
    chunk = await response.body_iterator.__anext__()

    assert "event.stream.gap" in chunk
    assert evicted_id in chunk

    await response.body_iterator.aclose()
