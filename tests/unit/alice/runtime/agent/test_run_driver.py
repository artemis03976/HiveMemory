from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.events import QueueFrameEventSink
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.runtime.agent.run_driver import RunDriver
from hivememory.core.mtp import MTPCallRequest, MTPCallResponse, MTPResponseStatus


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


@pytest.mark.asyncio
async def test_run_driver_applies_each_call_record_once():
    calls = 0
    applied = []

    async def run_frame(_frame, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="task"),
                suspend_action_id="act-1",
            )
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    async def resolve_call(_frame, _suspension, **_kwargs):
        return MTPCallResponse(status=MTPResponseStatus.SUCCESS, agent_alias="helper")

    runtime = SimpleNamespace(
        run_frame=run_frame,
        apply_call_response=lambda frame, suspension, response: applied.append(
            (frame, suspension, response)
        ),
    )
    driver = RunDriver(runtime, SimpleNamespace(resolve_call=resolve_call))

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(frame_id=""))
    result = await driver.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert len(applied) == 1
    assert driver.call_records[("", "act-1")].status.value == "applied"


@pytest.mark.asyncio
async def test_run_driver_drops_late_call_response_after_cancel():
    cancel_event = asyncio.Event()
    applied = []

    async def run_frame(_frame, **_kwargs):
        if cancel_event.is_set():
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="task"),
            suspend_action_id="act-1",
        )

    async def resolve_call(_frame, _suspension, **_kwargs):
        cancel_event.set()
        return MTPCallResponse(status=MTPResponseStatus.SUCCESS, agent_alias="helper")

    runtime = SimpleNamespace(
        run_frame=run_frame,
        apply_call_response=lambda *args: applied.append(args),
    )
    driver = RunDriver(runtime, SimpleNamespace(resolve_call=resolve_call))

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(frame_id=""))
    result = await driver.run(frame, cancel_event=cancel_event)

    assert result.status == FrameExecutionStatus.CANCELLED
    assert applied == []
    assert driver.call_records[("", "act-1")].status.value == "cancelled"


@pytest.mark.asyncio
async def test_run_driver_finalizes_root_run_exactly_once():
    finalize_run = MagicMock()

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime = SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run)
    driver = RunDriver(runtime)
    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))

    result = await driver.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    finalize_run.assert_called_once_with("run-1", result)
