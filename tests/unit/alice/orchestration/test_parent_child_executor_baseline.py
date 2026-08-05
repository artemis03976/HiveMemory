from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.output import NullFrameOutputSink
from hivememory.agent_runtime.products import FrameProducts
from hivememory.alice.orchestration.run_executor import RunExecutor
from hivememory.alice.orchestration.run_output import CallOutputFinished
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_context_provider import CallContext
from hivememory.alice.orchestration.sub_agent.call_coordinator import (
    CallCoordinator,
    CancelRun,
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.runtime.streaming import AgentRunStream
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope
from hivememory.core.mtp import MTPCallRequest, MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError, SystemFault


def _frame(frame_id: str = "frame-root") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-baseline", frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic-1",
        identity=Identity(user_id="user-1", agent_id="omni_doll"),
    )


def _suspension() -> FrameExecutionResult:
    return FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize"),
        suspend_action_id="action-1",
    )


def _session(frame: ExecutionFrame, *, cancel_event: asyncio.Event | None = None) -> RunSession:
    session = RunSession(
        agent_run_id=frame.runtime_scope.run_id,
        cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
    )
    session.register_root_frame(frame)
    return session


def _coordinator(runtime, child: ExecutionFrame, *, context_provider=None) -> CallCoordinator:
    if not hasattr(runtime, "apply_call_response"):
        runtime.apply_call_response = MagicMock()
    return CallCoordinator(
        runtime,
        context_provider
        or SimpleNamespace(
            provide=AsyncMock(return_value=CallContext(agent_profile=OMNI_DOLL_PROFILE))
        ),
        frame_factory=SimpleNamespace(
            scope=MagicMock(return_value=child.runtime_scope),
            create=MagicMock(return_value=child),
        ),
        prompt_assembler=SimpleNamespace(
            build_sub_agent_messages=MagicMock(return_value=child.working_history)
        ),
    )


class _RecordingRunOutput:
    def __init__(self) -> None:
        self.call_finished_outputs: list[CallOutputFinished] = []
        self._frame_output = NullFrameOutputSink()

    def for_frame(self, *_args, **_kwargs):
        return self._frame_output

    async def call_started(self, _output) -> None:
        return None

    async def call_finished(self, output: CallOutputFinished) -> None:
        self.call_finished_outputs.append(output)


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
async def test_root_terminal_outcomes_match_between_streaming_and_non_streaming(status):
    """流式与非流式入口必须共享 root 终态和 exactly-once 收尾语义。"""
    non_stream_frame = _frame("frame-non-stream")
    non_stream_finalize = MagicMock()
    non_stream_runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=status)),
        finalize_run=non_stream_finalize,
    )
    non_stream_executor = RunExecutor(
        non_stream_runtime,
        session=_session(non_stream_frame),
    )

    non_stream_result = await non_stream_executor.run(non_stream_frame)

    stream_frame = _frame("frame-stream")
    stream_finalize = MagicMock()
    stream_runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=status)),
        finalize_run=stream_finalize,
    )
    stream_session = _session(stream_frame)
    stream_executor = RunExecutor(stream_runtime, session=stream_session)
    agent_stream = AgentRunStream(stream_session)
    events = [
        event
        async for event in agent_stream.events(
            stream_executor.run(stream_frame, run_output=agent_stream.output)
        )
    ]

    assert non_stream_result.status == status
    assert stream_executor.terminal_result is not None
    assert stream_executor.terminal_result.status == status
    assert events == []
    non_stream_finalize.assert_called_once()
    stream_finalize.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_level", "expected_log"),
    [
        (
            PermissionDeniedError("denied"),
            logging.WARNING,
            "CALL target resolution rejected",
        ),
        (
            SystemFault("profile route unavailable"),
            logging.ERROR,
            "CALL context preparation failed",
        ),
    ],
)
async def test_call_context_error_is_returned_without_dispatching_child(
    error,
    expected_level,
    expected_log,
    caplog,
):
    caller = _frame()
    child = _frame("frame-child")
    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    context_provider = SimpleNamespace(provide=AsyncMock(side_effect=error))
    coordinator = _coordinator(runtime, child, context_provider=context_provider)
    output = _RecordingRunOutput()

    with caplog.at_level(logging.DEBUG):
        transition = await coordinator.begin_call(
            caller,
            _suspension(),
            session=_session(caller),
            run_output=output,
        )
    response = runtime.apply_call_response.call_args.args[2]

    assert transition == ResumeCaller()
    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == error.code
    runtime.run_frame.assert_not_awaited()
    runtime.finalize_frame.assert_not_called()
    assert len(output.call_finished_outputs) == 1
    assert output.call_finished_outputs[0].frame_id is None
    assert expected_log in caplog.text
    assert any(
        record.levelno == expected_level and expected_log in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_artifact_harvest_failure_becomes_stable_call_error_and_cleans_child():
    caller = _frame()
    child = _frame("frame-child")
    child_result = FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(return_value=child_result),
        finalize_frame=MagicMock(side_effect=[RuntimeError("harvest failed"), FrameProducts()]),
    )
    coordinator = _coordinator(runtime, child)

    suspension = _suspension()
    session = _session(caller)
    begin = await coordinator.begin_call(caller, suspension, session=session)
    assert begin == DispatchCallee(child)
    transition = await coordinator.complete_call(
        caller,
        suspension,
        child,
        child_result,
        session=session,
    )
    response = runtime.apply_call_response.call_args.args[2]

    assert transition == ResumeCaller()
    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.call_response.sub_agent_error"
    assert runtime.finalize_frame.call_count == 2
    cleanup_result = runtime.finalize_frame.call_args.args[1]
    assert cleanup_result.status == FrameExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_cancelled_session_stops_call_before_dispatch():
    caller = _frame()
    child = _frame("frame-child")
    cancel_event = asyncio.Event()
    cancel_event.set()

    async def run_frame(frame, *, cancel_event: asyncio.Event, **_kwargs):
        del frame
        assert cancel_event is _session_cancel_event
        assert cancel_event.is_set()
        return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

    _session_cancel_event = cancel_event
    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(side_effect=run_frame),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    coordinator = _coordinator(runtime, child)

    suspension = _suspension()
    session = _session(caller, cancel_event=cancel_event)
    begin = await coordinator.begin_call(
        caller,
        suspension,
        session=session,
    )
    assert begin == CancelRun()
    runtime.apply_call_response.assert_called_once()
    response = runtime.apply_call_response.call_args.args[2]
    assert response.status == MTPResponseStatus.CANCELLED
    runtime.run_frame.assert_not_awaited()
    runtime.finalize_frame.assert_not_called()
    assert session.call_records[("frame-root", "action-1")].status.value == "applied"


def test_run_executor_is_the_only_alice_run_frame_caller():
    """Alice 编排层仅递归 RunExecutor 可以推进 frame。"""
    repo_root = Path(__file__).resolve().parents[4]
    alice_root = repo_root / "src" / "hivememory" / "alice"
    callers: set[str] = set()

    for source_path in alice_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_frame"
            for node in ast.walk(tree)
        ):
            callers.add(source_path.name)

    assert callers == {"run_executor.py"}
