from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.products import FrameProducts
from hivememory.alice.runtime.agent.call_coordinator import (
    CallCoordinator,
    CallNextAction,
)
from hivememory.alice.runtime.agent.run_driver import RunDriver
from hivememory.alice.runtime.agent.run_session import RunSession
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope
from hivememory.core.mtp import MTPCallRequest, MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError


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


def _coordinator(runtime, child: ExecutionFrame, *, profile_resolver=None) -> CallCoordinator:
    if not hasattr(runtime, "apply_call_response"):
        runtime.apply_call_response = MagicMock()
    return CallCoordinator(
        runtime,
        profile_resolver or SimpleNamespace(resolve=AsyncMock(return_value=OMNI_DOLL_PROFILE)),
        SimpleNamespace(resolve=AsyncMock()),
        frame_factory=SimpleNamespace(
            scope=MagicMock(return_value=child.runtime_scope),
            create=MagicMock(return_value=child),
        ),
        prompt_assembler=SimpleNamespace(
            build_sub_agent_messages=MagicMock(return_value=child.working_history)
        ),
    )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    @property
    def wants_token_stream(self) -> bool:
        return False

    async def emit(self, event: dict) -> None:
        self.events.append(event)


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
    non_stream_driver = RunDriver(
        non_stream_runtime,
        session=_session(non_stream_frame),
    )

    non_stream_result = await non_stream_driver.run(non_stream_frame)

    stream_frame = _frame("frame-stream")
    stream_finalize = MagicMock()
    stream_runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=status)),
        finalize_run=stream_finalize,
    )
    stream_driver = RunDriver(stream_runtime, session=_session(stream_frame))
    events = [event async for event in stream_driver.run_stream(stream_frame)]

    assert non_stream_result.status == status
    assert stream_driver.terminal_result is not None
    assert stream_driver.terminal_result.status == status
    assert events == []
    non_stream_finalize.assert_called_once()
    stream_finalize.assert_called_once()


@pytest.mark.asyncio
async def test_call_preparation_error_is_returned_without_dispatching_child():
    caller = _frame()
    child = _frame("frame-child")
    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    profile_resolver = SimpleNamespace(
        resolve=AsyncMock(side_effect=PermissionDeniedError("denied"))
    )
    coordinator = _coordinator(runtime, child, profile_resolver=profile_resolver)
    sink = _RecordingSink()

    transition = await coordinator.begin_call(
        caller,
        _suspension(),
        session=_session(caller),
        event_sink=sink,
    )
    response = runtime.apply_call_response.call_args.args[2]

    assert transition.action == CallNextAction.RESUME_CALLER
    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.permission.denied"
    runtime.run_frame.assert_not_awaited()
    runtime.finalize_frame.assert_not_called()
    assert [event["event"] for event in sink.events] == ["sub_agent_end"]
    assert sink.events[0]["data"]["frame_id"] is None


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
    assert begin.action == CallNextAction.DISPATCH_CALLEE
    transition = await coordinator.complete_call(
        caller,
        suspension,
        child,
        child_result,
        session=session,
    )
    response = runtime.apply_call_response.call_args.args[2]

    assert transition.action == CallNextAction.RESUME_CALLER
    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.call_response.sub_agent_error"
    assert runtime.finalize_frame.call_count == 2
    cleanup_result = runtime.finalize_frame.call_args.args[1]
    assert cleanup_result.status == FrameExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_cancelled_session_reaches_child_with_same_cancel_event_before_dispatch():
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
    assert begin.action == CallNextAction.DISPATCH_CALLEE
    child_result = await runtime.run_frame(
        frame=child,
        generation_options=None,
        event_sink=SimpleNamespace(),
        cancel_event=cancel_event,
    )
    transition = await coordinator.complete_call(
        caller,
        suspension,
        child,
        child_result,
        session=session,
    )

    assert transition.action == CallNextAction.CANCEL_RUN
    runtime.apply_call_response.assert_not_called()
    runtime.run_frame.assert_awaited_once()
    runtime.finalize_frame.assert_called_once()


def test_run_scheduler_is_the_only_alice_run_frame_caller():
    """R3 永久门禁：Alice 编排层仅 RunScheduler 可以推进 frame。"""
    repo_root = Path(__file__).resolve().parents[5]
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

    assert callers == {"run_scheduler.py"}
