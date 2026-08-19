"""
CallCoordinator 错误路径集成测试 — 真实 Coordinator + 真实 RunSession 协作

验证 CallCoordinator 与 RunSession 两个真实内部组件在错误路径上的状态收口：
CALL 目标解析/上下文准备失败、artifact harvest 失败后的稳定错误码与 child 清理。
mock 仅限边界外端口（context_provider、runtime.run_frame/finalize_frame）。
"""

from __future__ import annotations

import logging
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
from hivememory.alice.orchestration.run_output import CallOutputFinished
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_context_provider import CallContext
from hivememory.alice.orchestration.sub_agent.call_coordinator import (
    CallCoordinator,
    DispatchCallee,
    ResumeCaller,
)
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.mtp import MTPCallRequest, MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError, SystemFault
from tests.helpers.workspace import make_runtime_scope


def _frame(frame_id: str = "frame-root") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=make_runtime_scope(run_id="run-baseline", frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic-1",
    )


def _suspension() -> FrameExecutionResult:
    return FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize"),
        suspend_action_id="action-1",
    )


def _session(frame: ExecutionFrame) -> RunSession:
    session = RunSession(agent_run_id=frame.runtime_scope.run_id)
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
