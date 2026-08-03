from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.events import QueueFrameEventSink
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.runtime.agent.call_coordinator import (
    CallNextAction,
    CallTransition,
)
from hivememory.alice.runtime.agent.run_driver import RunDriver
from hivememory.alice.runtime.agent.run_session import FrameSchedulingStatus, RunSession
from hivememory.core.mtp import MTPCallRequest


def _session_with(frame, *, cancel_event: asyncio.Event | None = None) -> RunSession:
    session = RunSession(
        agent_run_id=frame.runtime_scope.run_id,
        cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
    )
    session.register_root_frame(frame)
    return session


class _CallCoordinatorStub:
    def __init__(self, *, cancel_on_complete: bool = False, emit_end: bool = False) -> None:
        self.cancel_on_complete = cancel_on_complete
        self.emit_end = emit_end
        self.child = None
        self.callee_results = []

    async def begin_call(self, caller, suspension, *, session, **_kwargs):
        record = session.register_call(caller, suspension.suspend_action_id)
        record.begin_resolution()
        self.child = SimpleNamespace(
            runtime_scope=SimpleNamespace(run_id=session.agent_run_id, frame_id="frame-child")
        )
        session.register_callee_frame(self.child, record)
        return CallTransition(CallNextAction.DISPATCH_CALLEE, self.child)

    def event_sink_for_callee(self, _frame, _action_id, event_sink):
        return event_sink

    async def complete_call(
        self,
        caller,
        _suspension,
        callee,
        _result,
        *,
        session,
        event_sink,
        **_kwargs,
    ):
        self.callee_results.append(_result)
        record = session.call_for_callee(callee.runtime_scope.frame_id)
        record.mark_resolved()
        if self.emit_end:
            await event_sink.emit({"event": "sub_agent_end", "data": {"status": "success"}})
        if self.cancel_on_complete:
            session.cancel_event.set()
            record.cancel()
            return CallTransition(CallNextAction.CANCEL_RUN)
        record.mark_applied()
        return CallTransition(CallNextAction.RESUME_CALLER, caller)


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
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="task"),
                suspend_action_id="act-1",
            )
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub(emit_end=True)
    driver = RunDriver(
        SimpleNamespace(run_frame=run_frame, apply_call_response=MagicMock()),
        session=session,
        call_coordinator=coordinator,
    )
    events = [
        event
        async for event in driver.run_stream(
            frame,
            event_metadata={"agent_run_id": "run-1", "frame_id": "frame-1"},
        )
    ]

    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2, 3]
    assert [event["data"]["agent_run_id"] for event in events] == ["run-1"] * 4
    assert driver.next_stream_sequence == 4
    assert driver.terminal_result is not None
    assert driver.terminal_result.status == FrameExecutionStatus.COMPLETED
    assert session.frame_statuses == {
        "frame-1": FrameSchedulingStatus.TERMINATED,
        "frame-child": FrameSchedulingStatus.TERMINATED,
    }


@pytest.mark.asyncio
async def test_run_scheduler_routes_unexpected_callee_suspension_to_call_completion():
    root_calls = 0

    async def run_frame(frame, **_kwargs):
        nonlocal root_calls
        if frame.runtime_scope.frame_id == "frame-child":
            return FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED)
        root_calls += 1
        if root_calls == 1:
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="task"),
                suspend_action_id="act-1",
            )
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub()
    scheduler = RunDriver(
        SimpleNamespace(run_frame=run_frame),
        session=session,
        call_coordinator=coordinator,
    )

    result = await scheduler.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert [item.status for item in coordinator.callee_results] == [FrameExecutionStatus.SUSPENDED]
    assert root_calls == 2


@pytest.mark.asyncio
async def test_run_driver_cancels_runner_when_stream_consumer_closes():
    runner_cancelled = asyncio.Event()

    async def run_frame(_frame, *, event_sink, **_kwargs):
        try:
            await event_sink.emit({"event": "token", "data": {"content": "started"}})
            await asyncio.Event().wait()
        finally:
            runner_cancelled.set()

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    driver = RunDriver(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
    )
    stream = driver.run_stream(frame)

    first_event = await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(runner_cancelled.wait(), timeout=1)

    assert first_event["event"] == "token"


@pytest.mark.asyncio
async def test_run_driver_accepts_resumed_call_without_applying_response():
    calls = 0

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

    runtime = SimpleNamespace(
        run_frame=run_frame,
        apply_call_response=MagicMock(),
    )
    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    driver = RunDriver(
        runtime,
        session=_session_with(frame),
        call_coordinator=_CallCoordinatorStub(),
    )
    result = await driver.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    runtime.apply_call_response.assert_not_called()


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

    runtime = SimpleNamespace(
        run_frame=run_frame,
        apply_call_response=lambda *args: applied.append(args),
    )
    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    session = _session_with(frame, cancel_event=cancel_event)
    driver = RunDriver(
        runtime,
        session=session,
        call_coordinator=_CallCoordinatorStub(cancel_on_complete=True),
    )
    result = await driver.run(frame)

    assert result.status == FrameExecutionStatus.CANCELLED
    assert applied == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        FrameExecutionStatus.COMPLETED,
        FrameExecutionStatus.CANCELLED,
        FrameExecutionStatus.FAILED,
        FrameExecutionStatus.BUDGET_EXHAUSTED,
    ],
)
async def test_run_driver_finalizes_each_terminal_status_exactly_once(status):
    finalize_run = MagicMock()

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=status)

    runtime = SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run)
    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    driver = RunDriver(runtime, session=_session_with(frame))

    result = await driver.run(frame)

    assert result.status == status
    finalize_run.assert_called_once_with("run-1", result)


@pytest.mark.asyncio
async def test_run_driver_rejects_unsupported_frame_status():
    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(
            status=cast(FrameExecutionStatus, "waiting_input"),
        )

    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    driver = RunDriver(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
    )

    with pytest.raises(RuntimeError, match="unsupported root status: 'waiting_input'"):
        await driver.run(frame)


@pytest.mark.asyncio
async def test_run_driver_rejects_unregistered_frame():
    frame = SimpleNamespace(runtime_scope=SimpleNamespace(run_id="run-1", frame_id="frame-1"))
    session = RunSession(agent_run_id="run-1")
    driver = RunDriver(SimpleNamespace(run_frame=MagicMock()), session=session)

    with pytest.raises(ValueError, match="not registered"):
        await driver.run(frame)
