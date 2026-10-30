"""CallCoordinator 单元测试。

Coordinator 与其持有的 RunSession 调用记录属于同一内聚单元，均使用真实对象；
上下文 provider 与 runtime 的 frame 执行为替身；提示词与布局使用真实实现。
仅调用状态测试替代 frame factory，提示词测试保留真实帧创建。
"""

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
    FrameProducts,
)
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.call.context import CallContext
from hivememory.alice.orchestration.call.coordinator import (
    CallCoordinator,
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.orchestration.call.record import CallRecord, CallRecordStatus
from hivememory.alice.orchestration.call.response import sub_agent_budget_exhausted_info
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.orchestration.output import CallOutputFinished
from hivememory.alice.orchestration.session import RunSession
from hivememory.config.alice import KoakumaConfig, MTPPromptConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, TurnEvent
from hivememory.core.mtp import MTPCallRequest, MTPCallResponse, MTPResponseStatus
from hivememory.core.mtp.exceptions import (
    PermissionDeniedError,
    SystemFault,
)
from hivememory.i18n import get_default_language, set_default_language
from tests.helpers.call import StubOperationSubmitter, make_call_context, make_callee
from tests.helpers.outputs import NullAgentRunOutput, NullFrameOutputSink
from tests.helpers.workspace import make_runtime_scope


def _frame(
    action_id: str = "act-1",
    *,
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
) -> ExecutionFrame:
    frame = ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        submit_operation=StubOperationSubmitter(),
        runtime_scope=make_runtime_scope(
            agent_id=agent_id,
            run_id="run-1",
            frame_id="frame-1",
            workspace_id=workspace_id,
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
        language="zh",
    )
    frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            content="<CALL helper>",
            action_id=action_id,
            verb="CALL",
            targets=("helper",),
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
    return CallCoordinator(
        runtime,
        context_provider
        or SimpleNamespace(
            provide=AsyncMock(return_value=make_call_context(agent_profile=OMNI_DOLL_PROFILE))
        ),
        frame_factory=frame_factory,
        koakuma_config=KoakumaConfig(enabled=False),
    )


def _session(caller: ExecutionFrame) -> RunSession:
    session = RunSession(agent_run_id=caller.runtime_scope.run_id, process_id="process-1")
    session.register_root_frame(caller)
    return session


def _child() -> ExecutionFrame:
    child = _frame()
    child.runtime_scope = child.runtime_scope.model_copy(update={"frame_id": "frame-child"})
    return child


async def _prepare_prompt_frame(
    profile: AgentProfile,
    config: KoakumaConfig,
    *,
    shared_context: str = "",
) -> ExecutionFrame:
    """从公共 CALL 入口取得真实子帧，只替代已授权的上下文解析端口。"""
    caller = _frame()
    context = replace(
        make_call_context(
            agent_profile=profile,
            allowed_operations=frozenset(
                {WorkspaceOperation.MEMORY_WRITE, WorkspaceOperation.AGENT_CALL}
            ),
        ),
        shared_context=shared_context,
    )
    coordinator = CallCoordinator(
        SimpleNamespace(),
        SimpleNamespace(provide=AsyncMock(return_value=context)),
        frame_factory=FrameFactory(),
        koakuma_config=config,
    )
    result = await coordinator.begin_call(
        caller, _suspension(), session=_session(caller), run_output=NullAgentRunOutput()
    )
    assert isinstance(result, DispatchCallee)
    return result.frame


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_language", "persona_header", "teaching_intro"),
    [
        ("zh", "### 角色设定 ###", "你是运行在 HiveMemory 上的智能 Agent"),
        ("en", "### PERSONA ###", "You are an intelligent Agent running on HiveMemory"),
        ("", "### PERSONA ###", "You are an intelligent Agent running on HiveMemory"),
    ],
)
async def test_callee_prompt_uses_own_language_persona_and_policy(
    profile_language: str, persona_header: str, teaching_intro: str
) -> None:
    """子帧按自身 Profile 与操作快照组装，语言缺省时回退到进程默认值。"""
    previous = get_default_language()
    set_default_language("en")
    try:
        frame = await _prepare_prompt_frame(
            AgentProfile(persona="  子 Agent 人设\n", language=profile_language),
            KoakumaConfig(
                mtp_prompt=MTPPromptConfig(include_demo=False, include_error_handling=False)
            ),
            shared_context="\n[共享上下文]\n原文  ",
        )
    finally:
        set_default_language(previous)

    assert [message["role"] for message in frame.working_history] == ["system", "user"]
    system_prompt = frame.working_history[0]["content"]
    assert teaching_intro in system_prompt
    assert system_prompt.endswith(
        f"\n\n{persona_header}\n\n  子 Agent 人设\n\n\n\n[共享上下文]\n原文  "
    )
    assert frame.execution_policy.permitted_verbs == frozenset({"WRITE", "RUN"})
    # 子帧执行期的 MTP 回填沿用提示词使用的同一语言。
    assert frame.language == ("zh" if profile_language == "zh" else "en")
    assert "- WRITE:" in system_prompt
    assert "- RUN:" in system_prompt
    assert "CALL" not in system_prompt
    assert "READ" not in system_prompt
    assert frame.working_history[-1] == {"role": "user", "content": "summarize"}


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,teaching_enabled", [(False, True), (True, False)])
async def test_callee_prompt_without_teaching_keeps_persona_and_shared_context(
    enabled: bool, teaching_enabled: bool
) -> None:
    """任一教学开关关闭时仍保留子帧人设、共享上下文与任务。"""
    frame = await _prepare_prompt_frame(
        AgentProfile(persona="子 Agent", language="zh"),
        KoakumaConfig(enabled=enabled, mtp_prompt=MTPPromptConfig(enabled=teaching_enabled)),
        shared_context="[共享上下文]",
    )

    assert frame.working_history == [
        {"role": "system", "content": "### 角色设定 ###\n\n子 Agent\n\n[共享上下文]"},
        {"role": "user", "content": "summarize"},
    ]


@pytest.mark.asyncio
async def test_callee_prompt_omits_empty_system_message() -> None:
    """系统片段全部为空时只发送任务，不创建空系统消息。"""
    frame = await _prepare_prompt_frame(
        AgentProfile(persona="", language="zh"), KoakumaConfig(enabled=False)
    )

    assert frame.working_history == [{"role": "user", "content": "summarize"}]


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


def test_apply_call_response_is_exactly_once_and_records_outcome_on_result():
    """回填写出配对的结果事件：子帧结局与产物只在结果上，挂起时的调用事件保持原样。"""
    runtime = AgentRuntime(
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

    call = frame.progress.turn_events[0]
    runtime.apply_call_response(frame, suspension, response)

    assert [message["role"] for message in frame.working_history] == ["assistant", "user"]
    assert frame.progress.turn_events[0] == call
    result = frame.progress.turn_events[-1]
    assert (result.kind, result.sequence, result.action_id) == ("tool_result", 1, "act-1")
    assert (result.status, result.produced_aliases) == ("success", ("draft-1",))
    assert frame.harvested_aliases == ["draft-1"]
    with pytest.raises(ValueError, match="already applied"):
        runtime.apply_call_response(frame, suspension, response)


def test_apply_call_response_formats_with_frame_language_not_profile_language():
    """CALL 响应按帧的已解析语言回填；Profile 的语言设置不再参与执行期渲染。"""
    runtime = AgentRuntime(
        runtime_config=MagicMock(),
        loop_executor=MagicMock(),
    )
    frame = _frame()
    frame.language = "en"
    response = MTPCallResponse(
        status=MTPResponseStatus.ERROR,
        agent_alias="helper",
        error=sub_agent_budget_exhausted_info("helper"),
    )

    runtime.apply_call_response(frame, _suspension(), response)

    assert frame.agent_profile.language == "zh"
    assert "exhausted its iteration budget" in frame.working_history[-1]["content"]


@pytest.mark.parametrize("action_id", ["wrong", None])
def test_apply_call_response_rejects_wrong_or_missing_action(action_id):
    runtime = AgentRuntime(
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
    session = RunSession(agent_run_id="run-1", process_id="process-1")
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

    begin = await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )

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
        run_output=NullAgentRunOutput(),
    )

    assert complete == ResumeCaller()
    runtime.finalize_frame.assert_called_once()
    runtime.apply_call_response.assert_called_once()
    assert record.status == CallRecordStatus.APPLIED


@pytest.mark.asyncio
async def test_cancel_call_finalizes_bound_child_ends_its_thread_and_marks_record_once():
    """取消沿递归栈清理 CALL：收尾已绑定的子帧并结束子线程，重复取消是空操作。"""
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
    begin = await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )
    assert begin == DispatchCallee(child)

    coordinator.cancel_call(caller, suspension, session=session)
    coordinator.cancel_call(caller, suspension, session=session)

    runtime.finalize_frame.assert_called_once()
    assert runtime.finalize_frame.call_args.args[0] is child
    assert runtime.finalize_frame.call_args.args[1].status == FrameExecutionStatus.CANCELLED
    runtime.apply_call_response.assert_not_called()
    assert session.call_for_callee("frame-child").status == CallRecordStatus.CANCELLED
    assert child.submit_operation.end_calls == 1


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
    begin = await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )
    assert begin == DispatchCallee(child)
    transition = await coordinator.complete_call(
        caller, suspension, child, child_result, session=session, run_output=NullAgentRunOutput()
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
    # 子帧以终态结束即结束子线程；各终态在收尾入口共用同一次结束。
    assert child.submit_operation.end_calls == 1


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

    await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )
    transition = await coordinator.complete_call(
        caller, suspension, child, child_result, session=session, run_output=NullAgentRunOutput()
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
    await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )
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
    await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )

    with pytest.raises(ValueError, match="terminal callee result"):
        await coordinator.complete_call(
            caller,
            suspension,
            child,
            FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED),
            session=session,
            run_output=NullAgentRunOutput(),
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
            session=RunSession(agent_run_id="run-1", process_id="process-1"),
            run_output=NullAgentRunOutput(),
        )

    runtime.run_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_frame_holds_callee_thread_and_takes_its_labels() -> None:
    """子帧持有子线程的句柄，观测标签取子线程描述中被调用方的身份，不沿用调用方。"""
    caller = _frame(agent_id="registered_agent", workspace_id="isolation_workspace")
    callee = make_callee(AgentProfile(agent_id="helper"))
    runtime = SimpleNamespace(
        max_iterations=8,
        apply_call_response=MagicMock(),
    )
    coordinator = CallCoordinator(
        runtime,
        SimpleNamespace(provide=AsyncMock(return_value=CallContext(callee=callee))),
        frame_factory=FrameFactory(),
        koakuma_config=KoakumaConfig(enabled=False),
    )
    session = _session(caller)

    result = await coordinator.begin_call(
        caller, _suspension(), session=session, run_output=NullAgentRunOutput()
    )

    assert isinstance(result, DispatchCallee)
    assert result.frame.submit_operation is callee
    assert result.frame.runtime_scope.labels == callee.thread.labels
    assert result.frame.runtime_scope.labels.agent_id == "helper"
    assert result.frame.runtime_scope.run_id == caller.runtime_scope.run_id
    assert result.frame.runtime_scope.frame_id != caller.runtime_scope.frame_id
    assert callee.end_calls == 0


def _failing_frame_factory(caller, callee):
    """子帧创建失败：派生已成功，子帧还没开始运行。"""
    return SimpleNamespace(
        scope=MagicMock(return_value=caller.runtime_scope),
        create=MagicMock(side_effect=RuntimeError("frame build failed")),
    )


def _rebinding_frame_factory(caller, callee):
    """创建出与调用方同一标识的子帧，会话绑定以编排不变量违规失败。"""
    frame = _frame()
    frame.submit_operation = callee
    return SimpleNamespace(
        scope=MagicMock(return_value=caller.runtime_scope),
        create=MagicMock(return_value=frame),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_factory, raises",
    [
        pytest.param(_failing_frame_factory, False, id="frame-creation"),
        pytest.param(_rebinding_frame_factory, True, id="session-binding"),
    ],
)
async def test_failure_after_derivation_before_child_runs_ends_child_thread(make_factory, raises):
    """派生成功之后、子帧开始运行之前失败：子帧不会运行，子线程随之结束。

    子帧创建失败以准备错误回填调用方；会话绑定失败属于编排不变量违规，原样抛出。
    """
    caller = _frame()
    callee = make_callee(AgentProfile(agent_id="helper"))
    runtime = SimpleNamespace(max_iterations=8, apply_call_response=MagicMock())
    coordinator = CallCoordinator(
        runtime,
        SimpleNamespace(provide=AsyncMock(return_value=CallContext(callee=callee))),
        frame_factory=make_factory(caller, callee),
        koakuma_config=KoakumaConfig(enabled=False),
    )
    session = _session(caller)

    if raises:
        with pytest.raises(RuntimeError, match="already exists"):
            await coordinator.begin_call(
                caller, _suspension(), session=session, run_output=NullAgentRunOutput()
            )
    else:
        result = await coordinator.begin_call(
            caller, _suspension(), session=session, run_output=NullAgentRunOutput()
        )
        assert result == ResumeCaller()
        assert runtime.apply_call_response.call_args.args[2].status == MTPResponseStatus.ERROR
    assert callee.end_calls == 1


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
    begin = await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )
    assert begin == DispatchCallee(child)
    transition = await coordinator.complete_call(
        caller, suspension, child, child_result, session=session, run_output=NullAgentRunOutput()
    )
    response = runtime.apply_call_response.call_args.args[2]

    assert transition == ResumeCaller()
    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.call_response.sub_agent_error"
    assert runtime.finalize_frame.call_count == 2
    cleanup_result = runtime.finalize_frame.call_args.args[1]
    assert cleanup_result.status == FrameExecutionStatus.FAILED
