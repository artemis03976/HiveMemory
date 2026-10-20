"""
RunExecutor 集成测试 — 真实执行器/会话/交互输出端口三组件协作

验证 RunExecutor + RunSession + SinkRunOutput 的真实协作：SUSPENDED 重入的
交互事件序号连续性、run 被取消时执行器与会话的协同清理。仅 stub LLM 执行
端口（run_frame）与 call_coordinator（_CallCoordinatorStub）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.interface.output import SinkRunOutput
from hivememory.alice.orchestration.call.coordinator import (
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.orchestration.executor import RunExecutor
from hivememory.alice.orchestration.output import CallOutputFinished
from hivememory.alice.orchestration.session import RunSession
from hivememory.core.mtp import MTPCallRequest
from tests.helpers.alice import CollectedEvents
from tests.helpers.workspace import make_runtime_scope


def _start(executor, frame, session) -> tuple[asyncio.Task, CollectedEvents]:
    """在独立 task 中运行执行器，交互事件经真实输出端口推送到收集用的 sink。"""
    events = CollectedEvents()
    output = SinkRunOutput(events)
    output.bind_run(session.agent_run_id)
    return asyncio.create_task(executor.run(frame, run_output=output)), events


def _session_with(frame) -> RunSession:
    session = RunSession(agent_run_id=frame.runtime_scope.run_id, process_id="process-1")
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
    run, collected = _start(executor, frame, session)
    await run
    events = collected.events

    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2, 3]
    assert [event["data"]["agent_run_id"] for event in events] == ["run-1"] * 4
    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.COMPLETED
    assert set(session.frames) == {"frame-1", "frame-child"}
    assert session.call_records[("frame-1", "act-1")].status.value == "applied"


@pytest.mark.asyncio
async def test_run_executor_cancellation_unwinds_runner_and_records_cancelled_terminal():
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
    run, events = _start(executor, frame, session)

    first_event = await events.wait_for(lambda event: event["event"] == "token")
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert first_event["data"]["content"] == "started"
    assert runner_cancelled.is_set()
    assert executor.terminal_result is not None
    assert executor.terminal_result.status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_executor_cancellation_cleans_active_call():
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
    run, events = _start(executor, frame, session)

    await events.wait_for(lambda event: event["data"].get("content") == "child")
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert [event["data"]["content"] for event in events.events] == ["root", "child"]
    assert coordinator.cancel_calls == 1
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"


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
    run, _events = _start(executor, frame, session)

    await asyncio.wait_for(preparation_started.wait(), timeout=1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert coordinator.cancel_calls == 1
    assert session.call_records[("frame-1", "act-1")].status.value == "cancelled"
