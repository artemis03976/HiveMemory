from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.orchestration.call_coordinator import (
    CallNextAction,
    CallTransition,
)
from hivememory.alice.orchestration.run_output import CallOutputFinished
from hivememory.alice.orchestration.run_scheduler import RunScheduler
from hivememory.alice.orchestration.run_session import FrameSchedulingStatus, RunSession
from hivememory.alice.runtime.streaming import AgentRunStream, QueueAgentRunOutput
from hivememory.core.mtp import MTPCallRequest


def _session_with(frame, *, cancel_event: asyncio.Event | None = None) -> RunSession:
    session = RunSession(
        agent_run_id=frame.runtime_scope.run_id,
        cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
    )
    session.register_root_frame(frame)
    return session


def _frame_stub(frame_id: str, *, run_id: str = "run-1"):
    return SimpleNamespace(
        runtime_scope=SimpleNamespace(run_id=run_id, frame_id=frame_id),
        agent_profile=SimpleNamespace(alias="helper" if frame_id != "frame-1" else "main"),
        identity=SimpleNamespace(agent_id="owner"),
    )


class _CallCoordinatorStub:
    def __init__(self, *, cancel_on_complete: bool = False, emit_end: bool = False) -> None:
        self.cancel_on_complete = cancel_on_complete
        self.emit_end = emit_end
        self.child = None
        self.callee_results = []
        self.cancel_calls = 0

    async def begin_call(self, caller, suspension, *, session, **_kwargs):
        record = session.register_call(caller, suspension.suspend_action_id)
        record.begin_resolution()
        self.child = _frame_stub("frame-child", run_id=session.agent_run_id)
        session.register_callee_frame(self.child, record)
        return CallTransition(CallNextAction.DISPATCH_CALLEE, self.child)

    async def complete_call(
        self,
        caller,
        _suspension,
        callee,
        _result,
        *,
        session,
        run_output,
        **_kwargs,
    ):
        self.callee_results.append(_result)
        record = session.call_for_callee(callee.runtime_scope.frame_id)
        record.mark_resolved()
        if self.emit_end:
            await run_output.call_finished(
                CallOutputFinished(
                    status="success",
                    final_text="",
                    iteration=1,
                    action_id="act-1",
                    frame_id=callee.runtime_scope.frame_id,
                    agent_id="helper",
                    terminal_status="completed",
                )
            )
        if self.cancel_on_complete:
            session.cancel_event.set()
            record.cancel()
            return CallTransition(CallNextAction.CANCEL_RUN)
        record.mark_applied()
        return CallTransition(CallNextAction.RESUME_CALLER, caller)

    def cancel_call(self, caller, suspension, *, session):
        self.cancel_calls += 1
        record = session.require_call(caller, suspension.suspend_action_id)
        record.cancel()


@pytest.mark.asyncio
async def test_queue_sink_applies_backpressure_at_capacity():
    output = QueueAgentRunOutput("run-1", maxsize=1)

    await output.send_event("token", {"content": "first"})
    second_emit = asyncio.create_task(output.send_event("token", {"content": "second"}))
    await asyncio.sleep(0)

    assert not second_emit.done()
    assert (await output.receive())["data"]["stream_sequence"] == 0

    await second_emit
    assert (await output.receive())["data"]["stream_sequence"] == 1


@pytest.mark.asyncio
async def test_run_scheduler_reenters_suspended_frame_with_continuous_stream_sequence():
    calls = 0

    async def run_frame(_frame, *, output_sink, **_kwargs):
        nonlocal calls
        calls += 1
        await output_sink.send(TokenDelta(content=str(calls)))
        if calls == 1:
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="task"),
                suspend_action_id="act-1",
            )
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub(emit_end=True)
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame, apply_call_response=MagicMock()),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    events = [
        event
        async for event in agent_stream.events(scheduler.run(frame, run_output=agent_stream.output))
    ]

    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2, 3]
    assert [event["data"]["agent_run_id"] for event in events] == ["run-1"] * 4
    assert agent_stream.next_sequence == 4
    assert scheduler.terminal_result is not None
    assert scheduler.terminal_result.status == FrameExecutionStatus.COMPLETED
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

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub()
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame),
        session=session,
        call_coordinator=coordinator,
    )

    result = await scheduler.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert [item.status for item in coordinator.callee_results] == [FrameExecutionStatus.SUSPENDED]
    assert root_calls == 2


@pytest.mark.asyncio
async def test_run_scheduler_cancels_runner_when_stream_consumer_closes():
    runner_cancelled = asyncio.Event()

    async def run_frame(_frame, *, output_sink, **_kwargs):
        try:
            await output_sink.send(TokenDelta(content="started"))
            await asyncio.Event().wait()
        finally:
            runner_cancelled.set()

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame),
        session=session,
    )
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(scheduler.run(frame, run_output=agent_stream.output))

    first_event = await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(runner_cancelled.wait(), timeout=1)

    assert first_event["event"] == "token"


@pytest.mark.asyncio
async def test_run_scheduler_cleans_active_call_when_stream_consumer_closes():
    root_calls = 0

    async def run_frame(frame, *, output_sink, **_kwargs):
        nonlocal root_calls
        if frame.runtime_scope.frame_id == "frame-child":
            await output_sink.send(TokenDelta(content="child"))
            await asyncio.Event().wait()
        root_calls += 1
        await output_sink.send(TokenDelta(content="root"))
        return FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="task"),
            suspend_action_id="act-1",
        )

    finalize_run = MagicMock()
    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub()
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(scheduler.run(frame, run_output=agent_stream.output))

    assert (await anext(stream))["data"]["content"] == "root"
    assert (await anext(stream))["data"]["content"] == "child"
    await stream.aclose()

    assert coordinator.cancel_calls == 1
    assert session.cancel_event.is_set()
    assert session.active_frame_id is None
    assert session.frame_statuses == {
        "frame-1": FrameSchedulingStatus.TERMINATED,
        "frame-child": FrameSchedulingStatus.TERMINATED,
    }
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
    finalize_run.assert_called_once()
    assert finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_scheduler_cancels_record_during_call_preparation_await():
    preparation_started = asyncio.Event()

    class BlockingPreparationCoordinator(_CallCoordinatorStub):
        async def begin_call(self, caller, suspension, *, session, **_kwargs):
            record = session.register_call(caller, suspension.suspend_action_id)
            record.begin_resolution()
            preparation_started.set()
            await asyncio.Event().wait()

    async def run_frame(_frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="root"))
        return FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="task"),
            suspend_action_id="act-1",
        )

    finalize_run = MagicMock()
    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = BlockingPreparationCoordinator()
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(scheduler.run(frame, run_output=agent_stream.output))

    await anext(stream)
    await asyncio.wait_for(preparation_started.wait(), timeout=1)
    await stream.aclose()

    assert coordinator.cancel_calls == 1
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
    assert session.frame_statuses == {"frame-1": FrameSchedulingStatus.TERMINATED}
    finalize_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_scheduler_accepts_resumed_call_without_applying_response():
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
    frame = _frame_stub("frame-1")
    scheduler = RunScheduler(
        runtime,
        session=_session_with(frame),
        call_coordinator=_CallCoordinatorStub(),
    )
    result = await scheduler.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    runtime.apply_call_response.assert_not_called()


@pytest.mark.asyncio
async def test_run_scheduler_drops_late_call_response_after_cancel():
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
    frame = _frame_stub("frame-1")
    session = _session_with(frame, cancel_event=cancel_event)
    scheduler = RunScheduler(
        runtime,
        session=session,
        call_coordinator=_CallCoordinatorStub(cancel_on_complete=True),
    )
    result = await scheduler.run(frame)

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
async def test_run_scheduler_finalizes_each_terminal_status_exactly_once(status):
    finalize_run = MagicMock()

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=status)

    runtime = SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run)
    frame = _frame_stub("frame-1")
    scheduler = RunScheduler(runtime, session=_session_with(frame))

    result = await scheduler.run(frame)

    assert result.status == status
    finalize_run.assert_called_once_with("run-1", result)


@pytest.mark.asyncio
async def test_run_scheduler_rejects_unsupported_frame_status():
    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(
            status=cast(FrameExecutionStatus, "waiting_input"),
        )

    frame = _frame_stub("frame-1")
    scheduler = RunScheduler(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
    )

    with pytest.raises(RuntimeError, match="unsupported root status: 'waiting_input'"):
        await scheduler.run(frame)


@pytest.mark.asyncio
async def test_run_scheduler_rejects_unregistered_frame():
    frame = _frame_stub("frame-1")
    session = RunSession(agent_run_id="run-1")
    scheduler = RunScheduler(SimpleNamespace(run_frame=MagicMock()), session=session)

    with pytest.raises(ValueError, match="not registered"):
        await scheduler.run(frame)
