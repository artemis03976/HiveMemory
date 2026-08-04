from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import NullFrameOutputSink, TokenDelta
from hivememory.alice.orchestration.run_executor import RunExecutor
from hivememory.alice.orchestration.run_output import CallOutputFinished
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_coordinator import (
    CancelRun,
    DispatchCallee,
    ResumeCaller,
)
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
    def __init__(
        self,
        *,
        cancel_on_complete: bool = False,
        emit_end: bool = False,
        child_ids: dict[str, str] | None = None,
    ) -> None:
        self.cancel_on_complete = cancel_on_complete
        self.emit_end = emit_end
        self.child_ids = child_ids or {}
        self.callee_results = []
        self.completed_callees: list[str] = []
        self.cancel_calls = 0

    async def begin_call(self, caller, suspension, *, session, **_kwargs):
        record = session.register_call(caller, suspension.suspend_action_id)
        record.begin_resolution()
        child_id = self.child_ids.get(suspension.suspend_action_id, "frame-child")
        child = _frame_stub(child_id, run_id=session.agent_run_id)
        session.register_callee_frame(child, record)
        return DispatchCallee(child)

    async def complete_call(
        self,
        caller,
        _suspension,
        callee,
        result,
        *,
        session,
        run_output,
        **_kwargs,
    ):
        self.callee_results.append(result)
        self.completed_callees.append(callee.runtime_scope.frame_id)
        record = session.call_for_callee(callee.runtime_scope.frame_id)
        record.mark_resolved()
        if self.emit_end:
            await run_output.call_finished(
                CallOutputFinished(
                    status="success",
                    final_text="",
                    iteration=1,
                    action_id=record.action_id,
                    frame_id=callee.runtime_scope.frame_id,
                    agent_id="helper",
                    terminal_status="completed",
                )
            )
        if self.cancel_on_complete:
            session.cancel_event.set()
            record.cancel()
            return CancelRun()
        record.mark_applied()
        return ResumeCaller()

    def cancel_call(self, caller, suspension, *, session):
        self.cancel_calls += 1
        record = session.require_call(caller, suspension.suspend_action_id)
        record.cancel()


class _RecordingRunOutput:
    def __init__(self) -> None:
        self.bindings: list[tuple[str, str | None, str, int]] = []
        self._frame_output = NullFrameOutputSink()

    def for_frame(self, frame, *, action_id, scope, depth):
        self.bindings.append((frame.runtime_scope.frame_id, action_id, scope, depth))
        return self._frame_output

    async def call_started(self, _output) -> None:
        return None

    async def call_finished(self, _output) -> None:
        return None


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
async def test_run_executor_reenters_suspended_frame_with_continuous_stream_sequence():
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
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, apply_call_response=MagicMock()),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    events = [
        event
        async for event in agent_stream.events(executor.run(frame, run_output=agent_stream.output))
    ]

    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2, 3]
    assert [event["data"]["agent_run_id"] for event in events] == ["run-1"] * 4
    assert agent_stream.next_sequence == 4
    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.COMPLETED
    assert set(session.frames) == {"frame-1", "frame-child"}
    assert session.call_records[("frame-1", "act-1")].status.value == "applied"


@pytest.mark.asyncio
async def test_run_executor_recursively_executes_nested_call_frames():
    frame_calls: dict[str, int] = {}

    async def run_frame(frame, **_kwargs):
        frame_id = frame.runtime_scope.frame_id
        frame_calls[frame_id] = frame_calls.get(frame_id, 0) + 1
        if frame_id == "frame-1" and frame_calls[frame_id] == 1:
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="outer"),
                suspend_action_id="act-root",
            )
        if frame_id == "frame-child" and frame_calls[frame_id] == 1:
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="nested", task="inner"),
                suspend_action_id="act-child",
            )
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    output = _RecordingRunOutput()
    coordinator = _CallCoordinatorStub(
        child_ids={
            "act-root": "frame-child",
            "act-child": "frame-grandchild",
        }
    )
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame),
        session=session,
        call_coordinator=coordinator,
    )

    result = await executor.run(frame, run_output=output)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert frame_calls == {"frame-1": 2, "frame-child": 2, "frame-grandchild": 1}
    assert coordinator.completed_callees == ["frame-grandchild", "frame-child"]
    assert output.bindings == [
        ("frame-1", None, "main", 0),
        ("frame-child", "act-root", "sub", 1),
        ("frame-grandchild", "act-child", "sub", 2),
    ]


@pytest.mark.asyncio
async def test_run_executor_cancellation_unwinds_each_nested_call_once():
    grandchild_started = asyncio.Event()
    finalize_run = MagicMock()

    async def run_frame(frame, **_kwargs):
        frame_id = frame.runtime_scope.frame_id
        if frame_id == "frame-1":
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="helper", task="outer"),
                suspend_action_id="act-root",
            )
        if frame_id == "frame-child":
            return FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="nested", task="inner"),
                suspend_action_id="act-child",
            )
        grandchild_started.set()
        await asyncio.Event().wait()

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = _CallCoordinatorStub(
        child_ids={
            "act-root": "frame-child",
            "act-child": "frame-grandchild",
        }
    )
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
        call_coordinator=coordinator,
    )
    task = asyncio.create_task(executor.run(frame))

    await asyncio.wait_for(grandchild_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert coordinator.cancel_calls == 2
    assert session.call_records[("frame-1", "act-root")].status.value == "cancelled"
    assert session.call_records[("frame-child", "act-child")].status.value == "cancelled"
    finalize_run.assert_called_once()
    assert finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_cancels_runner_when_stream_consumer_closes():
    runner_cancelled = asyncio.Event()

    async def run_frame(_frame, *, output_sink, **_kwargs):
        try:
            await output_sink.send(TokenDelta(content="started"))
            await asyncio.Event().wait()
        finally:
            runner_cancelled.set()

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    executor = RunExecutor(SimpleNamespace(run_frame=run_frame), session=session)
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(executor.run(frame, run_output=agent_stream.output))

    first_event = await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(runner_cancelled.wait(), timeout=1)

    assert first_event["event"] == "token"
    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_cleans_active_call_when_stream_consumer_closes():
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
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(executor.run(frame, run_output=agent_stream.output))

    assert (await anext(stream))["data"]["content"] == "root"
    assert (await anext(stream))["data"]["content"] == "child"
    await stream.aclose()

    assert coordinator.cancel_calls == 1
    assert session.cancel_event.is_set()
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
    finalize_run.assert_called_once()
    assert finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_cancels_record_during_call_preparation_await():
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
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
        call_coordinator=coordinator,
    )
    agent_stream = AgentRunStream(session)
    stream = agent_stream.events(executor.run(frame, run_output=agent_stream.output))

    await anext(stream)
    await asyncio.wait_for(preparation_started.wait(), timeout=1)
    await stream.aclose()

    assert coordinator.cancel_calls == 1
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
    finalize_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_executor_accepts_call_resume_without_applying_response_itself():
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

    runtime = SimpleNamespace(run_frame=run_frame, apply_call_response=MagicMock())
    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        runtime,
        session=_session_with(frame),
        call_coordinator=_CallCoordinatorStub(),
    )

    result = await executor.run(frame)

    assert result.status == FrameExecutionStatus.COMPLETED
    runtime.apply_call_response.assert_not_called()


@pytest.mark.asyncio
async def test_run_executor_drops_late_call_response_after_cancel():
    cancel_event = asyncio.Event()
    applied = []

    async def run_frame(current_frame, **_kwargs):
        if current_frame.runtime_scope.frame_id == "frame-child":
            return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
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
    executor = RunExecutor(
        runtime,
        session=session,
        call_coordinator=_CallCoordinatorStub(cancel_on_complete=True),
    )

    result = await executor.run(frame)

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
async def test_run_executor_finalizes_each_terminal_status_exactly_once(status):
    finalize_run = MagicMock()

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=status)

    runtime = SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run)
    frame = _frame_stub("frame-1")
    executor = RunExecutor(runtime, session=_session_with(frame))

    result = await executor.run(frame)

    assert result.status == status
    finalize_run.assert_called_once_with("run-1", result)


@pytest.mark.asyncio
async def test_run_executor_rejects_unsupported_frame_status():
    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=cast(FrameExecutionStatus, "waiting_input"))

    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
    )

    with pytest.raises(RuntimeError, match="unsupported frame status: 'waiting_input'"):
        await executor.run(frame)


@pytest.mark.asyncio
async def test_run_executor_rejects_malformed_call_before_entering_coordinator():
    coordinator = SimpleNamespace(begin_call=MagicMock())
    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED)
        )
    )
    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        runtime,
        session=_session_with(frame),
        call_coordinator=coordinator,
    )

    with pytest.raises(RuntimeError, match="missing its request"):
        await executor.run(frame)

    coordinator.begin_call.assert_not_called()


@pytest.mark.asyncio
async def test_run_executor_rejects_unregistered_frame():
    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        SimpleNamespace(run_frame=MagicMock()),
        session=RunSession(agent_run_id="run-1"),
    )

    with pytest.raises(ValueError, match="not registered"):
        await executor.run(frame)


@pytest.mark.asyncio
async def test_run_executor_rejects_a_second_run():
    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
    )

    await executor.run(frame)
    with pytest.raises(RuntimeError, match="more than once"):
        await executor.run(frame)
