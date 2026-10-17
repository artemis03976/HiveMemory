from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.context_budget import TurnContextBudgetExceededError
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import NullFrameOutputSink
from hivememory.alice.events import BoundAgentRunEvents
from hivememory.alice.orchestration.call.coordinator import (
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.orchestration.executor import RunExecutor
from hivememory.alice.orchestration.output import CallOutputFinished
from hivememory.alice.orchestration.session import RunSession
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.mtp import MTPCallRequest
from tests.helpers.workspace import make_runtime_scope


def _session_with(frame) -> RunSession:
    session = RunSession(agent_run_id=frame.runtime_scope.run_id)
    session.register_root_frame(frame)
    return session


def _frame_stub(frame_id: str, *, run_id: str = "run-1"):
    return SimpleNamespace(
        runtime_scope=make_runtime_scope(run_id=run_id, frame_id=frame_id),
        agent_profile=SimpleNamespace(agent_id="helper" if frame_id != "frame-1" else "main"),
    )


class _CallCoordinatorStub:
    def __init__(
        self,
        *,
        emit_end: bool = False,
        child_ids: dict[str, str] | None = None,
    ) -> None:
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


@pytest.mark.asyncio
async def test_run_executor_preserves_cancellation_when_run_cleanup_fails():
    started = asyncio.Event()

    async def run_frame(_frame, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    finalize_run = MagicMock(side_effect=RuntimeError("run cleanup failed"))
    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=_session_with(frame),
    )
    task = asyncio.create_task(executor.run(frame))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_still_finalizes_when_call_record_cleanup_fails():
    started = asyncio.Event()

    async def run_frame(_frame, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    session.cancel_unapplied_calls = MagicMock(side_effect=RuntimeError("record cleanup failed"))
    finalize_run = MagicMock()
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=finalize_run),
        session=session,
    )
    task = asyncio.create_task(executor.run(frame))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_preserves_cancellation_when_call_cleanup_fails():
    preparation_started = asyncio.Event()

    class FailingCancellationCoordinator(_CallCoordinatorStub):
        async def begin_call(self, caller, suspension, *, session, **_kwargs):
            record = session.register_call(caller, suspension.suspend_action_id)
            record.begin_resolution()
            preparation_started.set()
            await asyncio.Event().wait()

        def cancel_call(self, caller, suspension, *, session):
            self.cancel_calls += 1
            raise RuntimeError("call cleanup failed")

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="task"),
            suspend_action_id="act-1",
        )

    frame = _frame_stub("frame-1")
    session = _session_with(frame)
    coordinator = FailingCancellationCoordinator()
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame, finalize_run=MagicMock()),
        session=session,
        call_coordinator=coordinator,
    )
    task = asyncio.create_task(executor.run(frame))

    await preparation_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert coordinator.cancel_calls == 1
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.CANCELLED


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
@pytest.mark.parametrize(
    "status",
    [
        FrameExecutionStatus.COMPLETED,
        FrameExecutionStatus.CANCELLED,
        FrameExecutionStatus.FAILED,
        FrameExecutionStatus.BUDGET_EXHAUSTED,
    ],
)
async def test_run_executor_records_each_terminal_status(status):
    """四种终态都属于显式接受集合，漏任一成员会落入 unsupported 分支。"""

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(status=status)

    runtime = SimpleNamespace(run_frame=run_frame)
    frame = _frame_stub("frame-1")
    executor = RunExecutor(runtime, session=_session_with(frame))

    await executor.run(frame)
    assert executor.terminal_result.status == status


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


@pytest.mark.asyncio
async def test_budget_failure_emits_frame_statistics_without_changing_terminal_result():
    """预算拦截只投影数值与帧坐标，观测不改变失败结果，也不泄露历史正文。"""

    async def run_frame(_frame, **_kwargs):
        return FrameExecutionResult(
            status=FrameExecutionStatus.FAILED,
            error=TurnContextBudgetExceededError(1001, 1000),
        )

    sink = RecordingRuntimeEventSink()
    frame = _frame_stub("frame-1")
    executor = RunExecutor(
        SimpleNamespace(run_frame=run_frame),
        session=_session_with(frame),
        run_events=BoundAgentRunEvents(RuntimeEventPublisher(sink)),
    )
    result = await executor.run(frame)
    assert result.status == FrameExecutionStatus.FAILED
    (event,) = sink.events
    assert event.event_type == "agent.context.budget_exceeded"
    assert event.frame_id == "frame-1"
    assert event.reason == "turn_context_budget_exceeded"
    assert event.data == {"estimated_tokens": 1001, "input_token_limit": 1000}
