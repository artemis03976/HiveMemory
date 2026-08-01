from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hivememory.agent_runtime.events import QueueFrameEventSink
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.runtime.agent.run_driver import RunDriver


@pytest.mark.asyncio
async def test_queue_sink_applies_backpressure_at_capacity():
    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=1)
    sink = QueueFrameEventSink(queue)

    await sink.emit({"event": "token", "data": {"content": "first"}})
    second_emit = asyncio.create_task(sink.emit({"event": "token", "data": {"content": "second"}}))
    await asyncio.sleep(0)

    assert not second_emit.done()
    assert (await queue.get())["data"]["stream_sequence"] == 0

    await second_emit
    assert (await queue.get())["data"]["stream_sequence"] == 1


@pytest.mark.asyncio
async def test_run_driver_reenters_suspended_frame_with_continuous_stream_sequence():
    calls = 0

    async def run_frame(_frame, *, event_sink, **_kwargs):
        nonlocal calls
        calls += 1
        await event_sink.emit({"event": "token", "data": {"content": str(calls)}})
        if calls == 1:
            return FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED)
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    async def on_suspend(_result, emit):
        await emit({"event": "sub_agent_end", "data": {"status": "success"}})

    driver = RunDriver(SimpleNamespace(run_frame=run_frame))
    events = [
        event
        async for event in driver.run_stream(
            object(),
            event_metadata={"agent_run_id": "run-1", "frame_id": "frame-1"},
            on_suspend=on_suspend,
        )
    ]

    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2]
    assert [event["data"]["agent_run_id"] for event in events] == ["run-1"] * 3
    assert driver.next_stream_sequence == 3
    assert driver.terminal_result is not None
    assert driver.terminal_result.status == FrameExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_driver_cancels_runner_when_stream_consumer_closes():
    runner_cancelled = asyncio.Event()

    async def run_frame(_frame, *, event_sink, **_kwargs):
        try:
            await event_sink.emit({"event": "token", "data": {"content": "started"}})
            await asyncio.Event().wait()
        finally:
            runner_cancelled.set()

    driver = RunDriver(SimpleNamespace(run_frame=run_frame))
    stream = driver.run_stream(object())

    first_event = await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(runner_cancelled.wait(), timeout=1)

    assert first_event["event"] == "token"
