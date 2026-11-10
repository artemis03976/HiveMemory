from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from sse_starlette.sse import AppStatus

from hivememory.components.events.bus import RuntimeEventBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.config.runtime import RuntimeEventsConfig
from hivememory.core.contracts.runtime_events import (
    RuntimeEvent,
    RuntimeEventType,
)
from hivememory.server.routers.runtime_events import runtime_events_status, stream_runtime_events


async def _next_chunk(response) -> str:
    """响应下一条事件按 SSE 线上格式编码后的文本。"""
    event = await response.body_iterator.__anext__()
    return event.encode().decode("utf-8")


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

    response = await stream_runtime_events(system=system)

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_runtime_events_stream_returns_sse_replay():
    bus = RuntimeEventBus()
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    response = await stream_runtime_events(system=system)
    chunk = await _next_chunk(response)

    assert response.media_type == "text/event-stream"
    assert "event: runtime_event" in chunk
    assert "chat.run.created" in chunk

    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_runtime_events_stream_preserves_workspace_correlations():
    """确认 SSE 重放保留观测所需的 Workspace 与进程标识投影。"""
    bus = RuntimeEventBus()
    bus.emit(
        RuntimeEvent(
            event_type=RuntimeEventType.CHAT_RUN_CREATED,
            process_id="process-isolated",
            workspace_id="isolation_workspace",
        )
    )
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    response = await stream_runtime_events(system=system)
    chunk = await _next_chunk(response)
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    payload = json.loads(data_line.removeprefix("data: "))

    assert payload["process_id"] == "process-isolated"
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

    response = await stream_runtime_events(
        last_event_id_query=first_id,
        system=system,
    )
    chunk = await _next_chunk(response)

    assert "chat.run.status" in chunk
    assert "streaming" in chunk
    assert first_id not in chunk

    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_runtime_events_stream_header_last_event_id_wins_over_stale_query():
    """自动重连沿用首次连接的 URL：请求头带最新收到的事件时按它续传，不重复推送。"""
    bus = RuntimeEventBus()
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_CREATED))
    stale_id = bus.buffer_snapshot[0].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status="streaming"))
    latest_seen_id = bus.buffer_snapshot[1].event_id
    bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_COMPLETED))
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    response = await stream_runtime_events(
        last_event_id_query=stale_id,
        last_event_id_header=latest_seen_id,
        system=system,
    )
    chunk = await _next_chunk(response)

    assert "chat.run.completed" in chunk
    assert "streaming" not in chunk

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

    response = await stream_runtime_events(
        last_event_id_header=evicted_id,
        system=system,
    )
    chunk = await _next_chunk(response)

    assert "event.stream.gap" in chunk
    assert evicted_id in chunk

    await response.body_iterator.aclose()


def _bus_with_statuses(*statuses: str) -> RuntimeEventBus:
    bus = RuntimeEventBus()
    for status in statuses:
        bus.emit(RuntimeEvent(event_type=RuntimeEventType.CHAT_RUN_STATUS, status=status))
    return bus


async def _replayed_statuses(response, count: int) -> list[str]:
    statuses = []
    for _ in range(count):
        chunk = await _next_chunk(response)
        data = json.loads(chunk.split("data: ", 1)[1])
        statuses.append(data["status"])
    await response.body_iterator.aclose()
    return statuses


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replay_last", "expected"),
    [(2, ["second", "third"]), (1000, ["first", "second", "third"])],
)
async def test_runtime_events_stream_replays_last_n_for_new_connection(replay_last, expected):
    """新连接按 replay_last 回放最近的事件，超过缓冲大小时回放整个缓冲。"""
    system = SimpleNamespace(
        runtime_events=_bus_with_statuses("first", "second", "third"),
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    response = await stream_runtime_events(replay_last=replay_last, system=system)

    assert await _replayed_statuses(response, len(expected)) == expected


@pytest.mark.asyncio
async def test_runtime_events_stream_resume_ignores_replay_last():
    """带回放点的续传回放其后的全部事件，不按 replay_last 截断。"""
    bus = _bus_with_statuses("first", "second", "third")
    first_id = bus.buffer_snapshot[0].event_id
    system = SimpleNamespace(
        runtime_events=bus,
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )

    response = await stream_runtime_events(
        last_event_id_header=first_id, replay_last=1, system=system
    )

    assert await _replayed_statuses(response, 2) == ["second", "third"]


@pytest.mark.asyncio
async def test_runtime_events_stream_ends_when_server_shuts_down(monkeypatch):
    """服务器开始关闭时流自行结束，即使客户端一直连着。

    服务器关闭时要先等进行中的请求结束，才执行应用的关闭流程；订阅没有尽头，
    不随关闭结束的流会让关闭停在等待连接上，话题也无法结算。
    """
    system = SimpleNamespace(
        runtime_events=RuntimeEventBus(),
        config=HiveMemoryConfig(runtime_events=RuntimeEventsConfig(enabled=True)),
    )
    response = await stream_runtime_events(system=system)
    monkeypatch.setattr(AppStatus, "should_exit", True)

    async def receive():
        # 客户端不发送断开消息：响应只能由服务器关闭结束。
        await asyncio.Event().wait()

    async def send(message):
        await asyncio.sleep(0)

    await asyncio.wait_for(response({"type": "http", "method": "GET"}, receive, send), timeout=2)
