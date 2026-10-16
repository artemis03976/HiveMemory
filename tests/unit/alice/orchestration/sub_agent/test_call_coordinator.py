"""CallCoordinator 单元测试。

Coordinator 与其持有的 RunSession 调用记录属于同一内聚单元，均使用真实对象；
context provider、frame factory、prompt assembler 与 runtime 的 frame 执行为替身。
"""

import asyncio
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
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.run_output import CallOutputFinished
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_coordinator import (
    CallCoordinator,
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.orchestration.sub_agent.call_record import CallRecord, CallRecordStatus
from hivememory.core.models import OMNI_DOLL_PROFILE, TurnEvent
from hivememory.core.mtp import MTPCallRequest, MTPCallResponse, MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError, SystemFault
from tests.helpers.call import make_call_context
from tests.helpers.workspace import make_runtime_scope


def _frame(
    action_id: str = "act-1",
    *,
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
) -> ExecutionFrame:
    frame = ExecutionFrame(
        runtime_scope=make_runtime_scope(
            agent_id=agent_id,
            run_id="run-1",
            frame_id="frame-1",
            workspace_id=workspace_id,
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
    )
    frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="<CALL helper>",
            action_id=action_id,
            tool_kind="CALL",
            tool_name="helper",
            status="pending",
        )
    )
    frame.progress.sequence = 1
    return frame


def _suspension(action_id: str = "act-1") -> FrameExecutionResult:
    return FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize"),
        suspend_assistant_text="<CALL helper>",
        suspend_action_id=action_id,
    )


def _coordinator(runtime, child: ExecutionFrame, *, context_provider=None) -> CallCoordinator:
    if not hasattr(runtime, "apply_call_response"):
        runtime.apply_call_response = MagicMock()
    frame_factory = SimpleNamespace(
        scope=MagicMock(return_value=child.runtime_scope),
        create=MagicMock(return_value=child),
    )
    prompt_assembler = SimpleNamespace(
        build_sub_agent_messages=MagicMock(return_value=child.working_history)
    )
    return CallCoordinator(
        runtime,
        context_provider
        or SimpleNamespace(
            provide=AsyncMock(return_value=make_call_context(agent_profile=OMNI_DOLL_PROFILE))
        ),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )


def _session(caller: ExecutionFrame) -> RunSession:
    session = RunSession(agent_run_id=caller.runtime_scope.run_id)
    session.register_frame(caller)
    return session


def _child() -> ExecutionFrame:
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    return child


class _RecordingRunOutput:
    """只记录 CALL 结束输出的 run output 替身。"""

    def __init__(self) -> None:
        self.call_finished_outputs: list[CallOutputFinished] = []
        self._frame_output = NullFrameOutputSink()

    def for_frame(self, *_args, **_kwargs):
        return self._frame_output

    async def call_started(self, _output) -> None:
        return None

    async def call_finished(self, output: CallOutputFinished) -> None:
        self.call_finished_outputs.append(output)


def test_apply_call_response_is_exactly_once_and_updates_call_pair():
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=MagicMock(),
    )
    frame = _frame()
    suspension = _suspension()
    response = MTPCallResponse(
        status=MTPResponseStatus.SUCCESS,
        agent_alias="helper",
        reply="done",
        artifact_aliases=["draft-1"],
    )

    runtime.apply_call_response(frame, suspension, response)

    assert [message["role"] for message in frame.working_history] == ["assistant", "user"]
    assert frame.progress.turn_events[0].status == "success"
    assert frame.progress.turn_events[-1].kind == "tool_result"
    assert frame.progress.turn_events[-1].action_id == "act-1"
    assert frame.harvested_aliases == ["draft-1"]
    with pytest.raises(ValueError, match="already applied"):
        runtime.apply_call_response(frame, suspension, response)


@pytest.mark.parametrize("action_id", ["wrong", None])
def test_apply_call_response_rejects_wrong_or_missing_action(action_id):
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=MagicMock(),
    )
    frame = _frame()
    suspension = _suspension(action_id) if action_id is not None else _suspension(None)
    response = MTPCallResponse(status=MTPResponseStatus.CANCELLED, agent_alias="helper")

    with pytest.raises(ValueError):
        runtime.apply_call_response(frame, suspension, response)


def test_call_record_cancel_wins_before_apply():
    record = CallRecord(caller_frame_id="frame-1", action_id="act-1")
    assert record.status == CallRecordStatus.SUSPENDED
    record.begin_resolution()
    record.mark_resolved()
    record.cancel()

    assert record.status == CallRecordStatus.CANCELLED
    with pytest.raises(RuntimeError, match="apply"):
        record.mark_applied()


def test_call_record_callee_binding_requires_resolving_and_is_exactly_once() -> None:
    session = RunSession(agent_run_id="run-1")
    root = _frame("act-1")
    callee = _frame("act-child")
    callee.runtime_scope = callee.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    session.register_root_frame(root)
    record = session.register_call(root, "act-1")

    with pytest.raises(RuntimeError, match="Cannot bind callee"):
        session.register_callee_frame(callee, record)

    record.begin_resolution()
    session.register_callee_frame(callee, record)
    with pytest.raises(RuntimeError, match="already has a callee frame"):
        record.bind_callee("frame-other")

    record.cancel()
    record.cancel()
    assert record.status.value == "cancelled"


@pytest.mark.asyncio
async def test_begin_and_complete_call_split_execution_from_coordination_phases():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    session = _session(caller)

    async def provide_context(*_args, **_kwargs):
        record = session.call_records[("frame-1", "act-1")]
        assert record.status == CallRecordStatus.RESOLVING
        return make_call_context(agent_profile=OMNI_DOLL_PROFILE)

    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(),
        finalize_frame=MagicMock(return_value=FrameProducts()),
        finalize_run=MagicMock(),
        apply_call_response=MagicMock(),
    )
    coordinator = _coordinator(
        runtime,
        child,
        context_provider=SimpleNamespace(provide=AsyncMock(side_effect=provide_context)),
    )
    suspension = _suspension()

    begin = await coordinator.begin_call(caller, suspension, session=session)

    assert begin == DispatchCallee(child)
    runtime.run_frame.assert_not_awaited()
    record = session.call_for_callee("frame-child")
    assert record.callee_frame_id == "frame-child"
    assert record.status == CallRecordStatus.RESOLVING

    complete = await coordinator.complete_call(
        caller,
        suspension,
        child,
        FrameExecutionResult(status=FrameExecutionStatus.COMPLETED),
        session=session,
    )

    assert complete == ResumeCaller()
    runtime.finalize_frame.assert_called_once()
    runtime.apply_call_response.assert_called_once()
    assert record.status == CallRecordStatus.APPLIED


@pytest.mark.asyncio
async def test_cancel_call_finalizes_bound_child_and_marks_record_once():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    session = _session(caller)
    runtime = SimpleNamespace(
        max_iterations=8,
        finalize_frame=MagicMock(return_value=FrameProducts()),
        apply_call_response=MagicMock(),
    )
    coordinator = _coordinator(runtime, child)
    suspension = _suspension()
    begin = await coordinator.begin_call(caller, suspension, session=session)
    assert begin == DispatchCallee(child)

    coordinator.cancel_call(caller, suspension, session=session)
    coordinator.cancel_call(caller, suspension, session=session)

    runtime.finalize_frame.assert_called_once()
    assert runtime.finalize_frame.call_args.args[0] is child
    assert runtime.finalize_frame.call_args.args[1].status == FrameExecutionStatus.CANCELLED
    runtime.apply_call_response.assert_not_called()
    assert session.call_for_callee("frame-child").status == CallRecordStatus.CANCELLED


@pytest.mark.asyncio
async def test_call_coordinator_finalizes_completed_child_without_finalizing_run():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    child.progress.text_segments.append("done")
    child_result = FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    runtime = SimpleNamespace(
        max_iterations=8,
        run_frame=AsyncMock(return_value=child_result),
        finalize_frame=MagicMock(return_value=FrameProducts(artifact_aliases=("draft-child",))),
        finalize_run=MagicMock(),
    )
    coordinator = _coordinator(runtime, child)
    session = _session(caller)

    suspension = _suspension()
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
    assert response.status == MTPResponseStatus.SUCCESS
    assert response.reply == "done"
    assert response.artifact_aliases == ["draft-child"]
    runtime.finalize_frame.assert_called_once_with(child, child_result)
    runtime.run_frame.assert_not_awaited()
    assert session.frames["frame-child"] is child
    assert session.call_for_callee("frame-child").status == CallRecordStatus.APPLIED


@pytest.mark.asyncio
async def test_cancelled_callee_resumes_caller_when_run_is_not_cancelled():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    child_result = FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
    runtime = SimpleNamespace(
        max_iterations=8,
        finalize_frame=MagicMock(return_value=FrameProducts()),
        apply_call_response=MagicMock(),
    )
    coordinator = _coordinator(runtime, child)
    session = _session(caller)
    suspension = _suspension()

    await coordinator.begin_call(caller, suspension, session=session)
    transition = await coordinator.complete_call(
        caller,
        suspension,
        child,
        child_result,
        session=session,
    )

    response = runtime.apply_call_response.call_args.args[2]
    assert transition == ResumeCaller()
    assert response.status == MTPResponseStatus.CANCELLED
    assert session.call_for_callee("frame-child").status == CallRecordStatus.APPLIED


@pytest.mark.asyncio
async def test_call_response_is_committed_before_finished_output_await():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    child_result = FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    runtime = SimpleNamespace(
        max_iterations=8,
        finalize_frame=MagicMock(return_value=FrameProducts()),
        apply_call_response=MagicMock(),
    )
    coordinator = _coordinator(runtime, child)
    session = _session(caller)
    suspension = _suspension()
    await coordinator.begin_call(caller, suspension, session=session)
    record = session.call_for_callee("frame-child")

    async def cancel_during_output(_output):
        runtime.apply_call_response.assert_called_once()
        assert record.status == CallRecordStatus.APPLIED
        raise asyncio.CancelledError

    output = SimpleNamespace(call_finished=AsyncMock(side_effect=cancel_during_output))
    with pytest.raises(asyncio.CancelledError):
        await coordinator.complete_call(
            caller,
            suspension,
            child,
            child_result,
            session=session,
            run_output=output,
        )

    coordinator.cancel_call(caller, suspension, session=session)
    runtime.apply_call_response.assert_called_once()


@pytest.mark.asyncio
async def test_preparation_error_is_committed_before_finished_output_await():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    runtime = SimpleNamespace(
        max_iterations=8,
        finalize_frame=MagicMock(return_value=FrameProducts()),
        apply_call_response=MagicMock(),
    )
    context_provider = SimpleNamespace(
        provide=AsyncMock(side_effect=RuntimeError("profile unavailable"))
    )
    coordinator = _coordinator(runtime, child, context_provider=context_provider)
    session = _session(caller)
    suspension = _suspension()

    async def cancel_during_output(_output):
        runtime.apply_call_response.assert_called_once()
        record = session.require_call(caller, "act-1")
        assert record.status == CallRecordStatus.APPLIED
        raise asyncio.CancelledError

    output = SimpleNamespace(call_finished=AsyncMock(side_effect=cancel_during_output))
    with pytest.raises(asyncio.CancelledError):
        await coordinator.begin_call(
            caller,
            suspension,
            session=session,
            run_output=output,
        )

    coordinator.cancel_call(caller, suspension, session=session)
    runtime.apply_call_response.assert_called_once()


@pytest.mark.asyncio
async def test_complete_call_rejects_a_nonterminal_callee_result():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    runtime = SimpleNamespace(
        max_iterations=8,
        finalize_frame=MagicMock(return_value=FrameProducts()),
        apply_call_response=MagicMock(),
    )
    coordinator = _coordinator(runtime, child)
    session = _session(caller)
    suspension = _suspension()
    await coordinator.begin_call(caller, suspension, session=session)

    with pytest.raises(ValueError, match="terminal callee result"):
        await coordinator.complete_call(
            caller,
            suspension,
            child,
            FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED),
            session=session,
        )

    runtime.finalize_frame.assert_not_called()
    runtime.apply_call_response.assert_not_called()


@pytest.mark.asyncio
async def test_call_coordinator_rejects_unregistered_caller():
    caller = _frame()
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    runtime = SimpleNamespace(run_frame=AsyncMock())
    coordinator = _coordinator(runtime, child)

    with pytest.raises(ValueError, match="not registered"):
        await coordinator.begin_call(
            caller,
            _suspension(),
            session=RunSession(agent_run_id="run-1"),
        )

    runtime.run_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_frame_inherits_caller_observation_labels() -> None:
    """防止 CALL 派生子帧时丢失注册标签或切回默认 Workspace 展示。"""
    caller = _frame(agent_id="registered_agent", workspace_id="isolation_workspace")
    runtime = SimpleNamespace(
        max_iterations=8,
        apply_call_response=MagicMock(),
    )
    coordinator = CallCoordinator(
        runtime,
        SimpleNamespace(
            provide=AsyncMock(return_value=make_call_context(agent_profile=OMNI_DOLL_PROFILE))
        ),
        frame_factory=FrameFactory(),
        prompt_assembler=SimpleNamespace(
            build_sub_agent_messages=MagicMock(
                return_value=[{"role": "user", "content": "summarize"}]
            )
        ),
    )
    session = _session(caller)

    result = await coordinator.begin_call(
        caller,
        _suspension(),
        session=session,
    )

    assert isinstance(result, DispatchCallee)
    assert result.frame.runtime_scope.labels == caller.runtime_scope.labels
    assert result.frame.runtime_scope.run_id == caller.runtime_scope.run_id
    assert result.frame.runtime_scope.frame_id != caller.runtime_scope.frame_id


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
    """目标解析被拒与上下文准备故障分别以对应级别记录，并回填稳定错误码且不派发子帧。"""
    caller = _frame()
    child = _child()
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
    assert any(
        record.levelno == expected_level and expected_log in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_artifact_harvest_failure_becomes_stable_call_error_and_cleans_child():
    """子帧产物收割失败时回填稳定 CALL 错误，并以 FAILED 再次收尾子帧。"""
    caller = _frame()
    child = _child()
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
