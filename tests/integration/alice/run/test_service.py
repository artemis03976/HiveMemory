"""
AgentRunService 集成测试 — 真实 AgentRuntime 装配链协作

驱动 AgentRunService、内核工厂装配的 AgentRuntime、CALL 编排、帧工厂、
提示词组装与真实事件管线；模型输出通过执行端口或继承 LLM 服务接口的替身提供。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.components.events.bus import (
    RecordingRuntimeEventSink,
)
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    AgentProfile,
)
from hivememory.core.mtp import MTP_STOP_SEQUENCE
from hivememory.infrastructure.llm.base import LLMCompletion, LLMRequestParams
from hivememory.workspace.contracts import CPUExecutionStatus
from tests.helpers.alice import CollectedEvents, build_run_service, run_with_events
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.llm import CallbackLLMService, ScriptedLLMService
from tests.helpers.outputs import NullRunOutput
from tests.helpers.sessions import make_conversation_turn
from tests.helpers.workspace import make_execution_labels


@pytest.mark.asyncio
async def test_session_model_and_output_override_control_budget_and_provider_request(
    tmp_path,
):
    """会话覆盖模型与输出上限后，预检预算和实际模型请求使用同一次解析。"""
    from hivememory.core.models.model_definition import ModelDefinition
    from hivememory.system.model_registry import ModelRegistry

    registry = ModelRegistry(registry_path=tmp_path / "models.yaml")
    registry.add_model(
        ModelDefinition(
            id="small",
            display_name="小窗口",
            litellm_model="provider/small",
            max_tokens=900,
            context_window=1000,
        )
    )
    config = HiveMemoryConfig()
    config.alice.koakuma.enabled = False
    config.alice.runtime.safety_margin = 100
    llm = ScriptedLLMService("完成")
    _, service = build_run_service(config=config, model_registry=registry, llm_service=llm)
    profile = AgentProfile(model_name="unavailable-profile-model", persona="", temperature=0.2)
    manifest = make_input_manifest(agent_profile=profile, user_message="x" * 2000)

    rejected = await service.run_agent(
        manifest, {"model": "small"}, run_output=NullRunOutput(), submit_operation=MagicMock()
    )
    assert rejected.failure_reason == "input_too_large"
    assert llm.calls == []

    completed = await service.run_agent(
        manifest,
        {"model": "small", "max_tokens": 100},
        run_output=NullRunOutput(),
        submit_operation=MagicMock(),
    )
    assert completed.status == "completed"
    assert completed.final_text == "完成"
    assert completed.model_used == "小窗口"
    # Alice 侧发给提供商的参数：解析后的模型与采样参数，外加 MTP 停止符。
    assert [call.params for call in llm.calls] == [
        LLMRequestParams(
            model="provider/small",
            temperature=0.2,
            top_p=1.0,
            max_tokens=100,
            stop=(MTP_STOP_SEQUENCE,),
        )
    ]


@pytest.mark.asyncio
async def test_call_child_guard_uses_callee_model_and_returns_error_to_parent(tmp_path):
    """子帧按自己的小窗口被拦截，父帧收到 CALL 错误后继续完成。"""
    from hivememory.core.models.model_definition import ModelDefinition
    from hivememory.system.model_registry import ModelRegistry
    from tests.helpers.call import StubOperationSubmitter, make_callee

    registry = ModelRegistry(registry_path=tmp_path / "models.yaml")
    for name, window in [("parent", 10000), ("child", 200)]:
        registry.add_model(
            ModelDefinition(
                id=name,
                display_name=name,
                litellm_model=f"provider/{name}",
                context_window=window,
                max_tokens=100,
            )
        )
    config = HiveMemoryConfig()
    config.alice.koakuma.mtp_prompt.enabled = False
    config.alice.runtime.safety_margin = 0
    recorder = RecordingRuntimeEventSink()
    llm = ScriptedLLMService('⟪ CALL | helper | task="总结" ⟫', "父帧完成")
    _, service = build_run_service(
        config=config, model_registry=registry, runtime_events=recorder, llm_service=llm
    )

    async def derive(_request):
        return make_callee(AgentProfile(model_name="child", persona="x" * 1000))

    result = await service.run_agent(
        make_input_manifest(
            agent_profile=AgentProfile(model_name="parent", persona=""),
            allowed_operations=frozenset({WorkspaceOperation.AGENT_CALL}),
        ),
        run_output=NullRunOutput(),
        submit_operation=StubOperationSubmitter(derive=derive),
    )
    assert (result.status, result.final_text) == ("completed", "父帧完成")
    call_result = next(e for e in result.turn_events if e.kind == "tool_result")
    assert call_result.status == "error"
    assert 'code="mtp.call_response.sub_agent_error"' in call_result.content
    assert [call.params.model for call in llm.calls] == ["provider/parent", "provider/parent"]
    (guard,) = [e for e in recorder.events if e.event_type == "agent.context.budget_exceeded"]
    assert guard.data["input_token_limit"] == 100
    assert recorder.events[-1].event_type == "agent.run.completed"


def _build_input_manifest(
    *,
    labels=None,
    agent_profile=None,
    instructions: str = "",
    process_id: str = "process-test",
    allowed_operations: frozenset[WorkspaceOperation] = frozenset(WorkspaceOperation),
):
    return make_input_manifest(
        labels=labels or make_execution_labels(agent_id="omni_doll"),
        agent_profile=agent_profile,
        instructions=instructions,
        allowed_operations=allowed_operations,
        process_id=process_id,
        topic_id="topic_1",
        user_message="hello",
        turn_context="ctx",
    )


def _stub_terminal_execution(
    runtime: AgentRuntime,
    status: FrameExecutionStatus = FrameExecutionStatus.COMPLETED,
) -> None:
    runtime.run_frame = AsyncMock(
        return_value=FrameExecutionResult(status=status),
    )


@pytest.mark.asyncio
async def test_run_agent_passes_input_and_submitter_to_root_frame():
    """CPU 交付的提交函数随 root frame 进入执行层，预检索内容仍进入提示词。"""
    runtime, service = build_run_service()
    context = _build_input_manifest().model_copy(
        update={
            "session_history": (
                make_conversation_turn(
                    agent_id="other", user_message="历史问题", assistant_final_text="历史回答"
                ),
            ),
        }
    )
    seen = []
    submit_operation = MagicMock()

    async def run_frame(frame, **_kwargs):
        seen.append(frame)
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime.run_frame = run_frame
    result = await service.run_agent(
        context, run_output=NullRunOutput(), submit_operation=submit_operation
    )
    assert result.status == CPUExecutionStatus.COMPLETED.value
    (frame,) = seen
    assert frame.submit_operation is submit_operation
    assert frame.working_history[-3:] == [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": "[From: other]\n历史回答"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_root_policy_and_prompt_follow_manifest_operations():
    """根 frame 的策略与教学由清单的操作快照翻译，系统工具菜单不受操作集合裁剪。"""
    runtime, service = build_run_service()
    manifest = _build_input_manifest(
        allowed_operations=frozenset(
            {WorkspaceOperation.MEMORY_READ, WorkspaceOperation.AGENT_CALL}
        ),
        agent_profile=AgentProfile(persona="执行人设"),
        instructions="### 角色设定 ###\n\n执行人设",
    )
    seen = []

    async def run_frame(frame, **_kwargs):
        seen.append(frame)
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    async def submit_operation(_request):
        raise AssertionError("本场景不应提交资源操作")

    runtime.run_frame = run_frame
    result = await service.run_agent(
        manifest, run_output=NullRunOutput(), submit_operation=submit_operation
    )
    assert result.status == "completed"

    (frame,) = seen
    assert frame.execution_policy.permitted_verbs == {"READ", "RUN", "CALL"}
    prompt = frame.working_history[0]["content"]
    assert "执行人设" in prompt
    assert "READ" in prompt and "CALL" in prompt
    assert "WRITE" not in prompt and "UPDATE" not in prompt and "SEARCH" not in prompt
    assert "sys_write_file" in prompt and "sys_python_repl" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_agent_id", "display_agent_id"),
    [("display_agent", "display_agent"), (None, "registered_agent")],
)
async def test_registered_labels_drive_runtime_events_and_profile_controls_stream_display(
    profile_agent_id, display_agent_id
) -> None:
    """真实执行链不携带授权身份，事件用注册标签，流展示优先 Profile 源 alias。"""
    from hivememory.core.models import AgentProfile

    async def chunks(_messages, _params):
        yield "标签保持一致"

    recorder = RecordingRuntimeEventSink()
    _runtime, service = build_run_service(
        runtime_events=recorder, llm_service=CallbackLLMService(stream=chunks)
    )
    manifest = _build_input_manifest(
        labels=make_execution_labels(
            agent_id="registered_agent", workspace_id="isolation_workspace"
        ),
        agent_profile=AgentProfile(agent_id=profile_agent_id),
        process_id="interaction-isolation",
    )

    result, events = await run_with_events(
        service, manifest, {"model": "test-model"}, submit_operation=MagicMock()
    )

    assert [event["event"] for event in events] == ["token"]
    assert [event["data"]["agent_id"] for event in events] == [display_agent_id]
    assert result.final_text == "标签保持一致"
    assert [event.event_type for event in recorder.events] == [
        RuntimeEventType.AGENT_RUN_STARTED,
        RuntimeEventType.AGENT_RUN_COMPLETED,
    ]
    assert [(event.agent_id, event.workspace_id) for event in recorder.events] == [
        ("registered_agent", "isolation_workspace"),
        ("registered_agent", "isolation_workspace"),
    ]


@pytest.mark.asyncio
async def test_run_agent_correlates_runtime_scope_and_process_id():
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest(process_id="process-1")
    _stub_terminal_execution(runtime)
    created_sessions = []
    create_run_session = service._create_run_session

    def _capture_session(**kwargs):
        session = create_run_session(**kwargs)
        created_sessions.append(session)
        return session

    service._create_run_session = _capture_session
    await service.run_agent(context, run_output=NullRunOutput(), submit_operation=MagicMock())

    session = created_sessions[0]
    assert session.process_id == "process-1"
    assert session.agent_run_id == recorder.events[0].agent_run_id
    assert recorder.events[0].process_id == "process-1"


@pytest.mark.asyncio
async def test_run_agent_failed_result_emits_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest()
    _stub_terminal_execution(runtime, FrameExecutionStatus.FAILED)

    result = await service.run_agent(
        context, run_output=NullRunOutput(), submit_operation=MagicMock()
    )

    assert result.status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].severity == "error"


@pytest.mark.asyncio
async def test_cancelled_run_returns_cancelled_result_and_publishes_cancelled_event():
    """原生取消在最外层转换为 cancelled 结果返回；agent.run.cancelled 的载荷保持关闭原因。"""
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest()

    async def _run_frame(_frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="hi"))
        await asyncio.Event().wait()

    runtime.run_frame = _run_frame
    events = CollectedEvents()
    run = asyncio.create_task(
        run_with_events(service, context, submit_operation=MagicMock(), events=events)
    )
    await events.wait_for(lambda event: event["event"] == "token")
    run.cancel()
    result, _ = await run

    assert result.status == CPUExecutionStatus.CANCELLED.value
    assert [event.event_type for event in recorder.events] == [
        RuntimeEventType.AGENT_RUN_STARTED,
        RuntimeEventType.AGENT_RUN_CANCELLED,
    ]
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"


@pytest.mark.asyncio
async def test_run_agent_error_publishes_failed_runtime_event():
    """普通异常不被转换为结果：照常抛出，并发布 agent.run.failed。"""
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest()
    runtime.run_frame = AsyncMock(side_effect=RuntimeError("network unavailable"))

    with pytest.raises(RuntimeError, match="network unavailable"):
        await run_with_events(service, context, submit_operation=MagicMock())

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"


@pytest.mark.asyncio
async def test_terminal_result_carries_turn_facts_and_event_keeps_iteration_stats():
    """终态结果携带回复、轮次事件与模型名；迭代统计只留在 agent.run.* 终态载荷中。"""
    from hivememory.agent_runtime.frame import FrameExecutionResult
    from hivememory.core.models import TurnEvent

    def _scripted_run_frame(frame, **_kwargs):
        async def _run():
            frame.progress.text_segments.append("hello ")
            frame.progress.text_segments.append("world")
            frame.progress.turn_events.append(
                TurnEvent(
                    kind="assistant_message",
                    sequence=frame.progress.sequence,
                    role="assistant",
                    content="hello world",
                )
            )
            frame.progress.sequence += 1
            frame.progress.iteration = 3
            frame.progress.model_used = "glm-4"
            return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

        return _run()

    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    runtime.run_frame = _scripted_run_frame

    result, _ = await run_with_events(
        service, _build_input_manifest(process_id="process-unified"), submit_operation=MagicMock()
    )

    assert result.status == CPUExecutionStatus.COMPLETED.value
    assert result.final_text == "hello world"
    assert [event.kind for event in result.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    assert result.model_used == "glm-4"
    assert [event.event_type for event in recorder.events] == [
        RuntimeEventType.AGENT_RUN_STARTED,
        RuntimeEventType.AGENT_RUN_COMPLETED,
    ]
    assert recorder.events[-1].data["mtp_iterations"] == 2
    assert recorder.events[-1].data["total_iterations"] == 3


@pytest.mark.asyncio
async def test_native_cancellation_returns_partial_text_and_completed_checkpoint():
    """真实 AgentLoop/RunExecutor 收口半截文字；取消后返回带部分输出与已完成检查点的结果。"""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def summarize(_messages, _params):
        return "Accepted summary"

    async def partial_answer(_messages, _params):
        try:
            yield "半截答案"
            entered.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    config, manifest = _summary_inputs()
    manifest = manifest.model_copy(update={"user_message": "hello"})
    _runtime, service = build_run_service(
        config=config, llm_service=CallbackLLMService(complete=summarize, stream=partial_answer)
    )
    run = asyncio.create_task(run_with_events(service, manifest, submit_operation=MagicMock()))
    await entered.wait()
    run.cancel()
    result, events = await run

    assert [event["data"].get("status", event["data"].get("content")) for event in events] == [
        "started",
        "completed",
        "半截答案",
    ]
    assert cancelled.is_set()
    assert result.status == "cancelled"
    assert result.final_text == "半截答案"
    assert result.compaction_checkpoint.summary == "Accepted summary"
    assert result.compaction_checkpoint.summarized_through == "t2"
    assert [(e.kind, e.content, e.status) for e in result.turn_events] == [
        ("user_message", "hello", None),
        ("assistant_message", "半截答案", "cancelled"),
    ]


@pytest.mark.asyncio
async def test_cancel_in_call_keeps_root_call_without_child_partial_text():
    """子帧中断时只保存根帧的不完整 CALL，不把尚未回填的子回复写成根回复。"""
    from tests.helpers.call import StubOperationSubmitter, make_callee

    calls = 0
    child_entered = asyncio.Event()

    async def generate_stream(_messages, _params):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield '⟪ CALL | helper | task="分析" ⟫'
        else:
            yield "子帧半截"
            child_entered.set()
            await asyncio.Event().wait()

    async def derive(_request):
        return make_callee(AgentProfile(agent_id="helper"))

    _runtime, service = build_run_service(llm_service=CallbackLLMService(stream=generate_stream))
    run = asyncio.create_task(
        run_with_events(
            service,
            _build_input_manifest(),
            submit_operation=StubOperationSubmitter(derive=derive),
        )
    )
    await child_entered.wait()
    run.cancel()
    result, _ = await run
    assert result.status == "cancelled"
    assert result.final_text == ""
    assert [(e.kind, e.tool_kind) for e in result.turn_events] == [
        ("user_message", None),
        ("tool_call", "CALL"),
    ]
    assert result.has_output is True


def _compaction_inputs(*, compressible=True):
    """小预算和超阈值旧历史，用真实渲染触发压缩。"""
    from hivememory.config.alice import CompactionConfig
    from hivememory.core.models import TurnEvent

    config = HiveMemoryConfig()
    config.alice.koakuma.enabled = False
    config.alice.runtime.default_context_window = 33768
    config.alice.runtime.safety_margin = 0
    config.alice.compaction = CompactionConfig(summary_max_tokens=64)
    turn = make_conversation_turn(
        interaction_id="old",
        user_message="old" if compressible else "x" * 6000,
        turn_events=(
            [
                TurnEvent(
                    kind="tool_result",
                    sequence=0,
                    role="user",
                    content="x" * 6000,
                    tool_kind="READ",
                    target="mem_a",
                    status="success",
                )
            ]
            if compressible
            else []
        ),
    )
    manifest = make_input_manifest(
        agent_profile=AgentProfile(persona="", language="en"),
        language="en",
        user_message="next",
        session_history=(turn,),
    )
    return config, manifest


@pytest.mark.asyncio
async def test_compaction_reaches_generation_and_returns_checkpoint_without_recording_progress():
    """减重不调用辅助模型；正常生成消费缩小后的视图，进度只作为交互事件推送、不进入轮次事件。"""

    async def complete(_messages, _params):
        return "完成"

    async def complete_stream(_messages, _params):
        yield "完成"

    llm = CallbackLLMService(complete=complete, stream=complete_stream)
    config, manifest = _compaction_inputs()
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder, llm_service=llm)
    result, events = await run_with_events(service, manifest, submit_operation=MagicMock())
    progress = [event for event in events if event["event"] == "context_compaction"]
    assert [event["data"]["status"] for event in progress] == ["started", "completed"]
    assert [event["data"]["stream_sequence"] for event in events] == list(range(len(events)))
    assert result.status == "completed"
    assert result.compaction_checkpoint.reduced_through == "old"
    assert [(event.kind, event.content) for event in result.turn_events] == [
        ("user_message", "next"),
        ("assistant_message", "完成"),
    ]
    (request,) = [call.messages for call in llm.calls]
    assert "READ again" in request[1]["content"]
    assert all("x" * 6000 not in message["content"] for message in request)
    observed = [
        event
        for event in recorder.events
        if event.event_type == RuntimeEventType.AGENT_CONTEXT_COMPACTION
    ]
    assert [event.status for event in observed] == ["started", "completed"]
    assert observed[-1].data["covered_turns"] == 1
    assert observed[-1].data["tokens_after"] < observed[-1].data["tokens_before"]
    assert set(observed[-1].data) == {
        "layer",
        "tokens_before",
        "tokens_after",
        "covered_turns",
        "elapsed_ms",
        "summary_status",
        "summarized_turns",
    }


@pytest.mark.asyncio
async def test_uncompressible_history_fails_without_output_and_keeps_reduction_checkpoint():
    """用户原文不能减重，仍超限时不调用生成模型；进度本身不构成输出。"""
    llm = ScriptedLLMService()
    config, manifest = _compaction_inputs(compressible=False)
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder, llm_service=llm)
    result, events = await run_with_events(service, manifest, submit_operation=MagicMock())
    assert [event["event"] for event in events] == [
        "context_compaction",
        "context_compaction",
    ]
    assert events[1]["data"]["status"] == "failed"
    assert result.failure_reason == "compaction_failed"
    assert result.has_output is False
    assert result.compaction_checkpoint.reduced_through == "old"
    assert llm.calls == []
    failed_compaction = [
        event
        for event in recorder.events
        if event.event_type == RuntimeEventType.AGENT_CONTEXT_COMPACTION
        and event.status == "failed"
    ]
    assert [event.reason for event in failed_compaction] == ["summary_input_too_large"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_at", ["started", "completed"])
async def test_cancel_at_compaction_progress_returns_only_completed_checkpoint(cancelled_at):
    """推送压缩开始进度时被取消没有可交回的结果，取消原样传播；推送完成进度时被取消
    以 cancelled 结果交回已完成的视图。"""
    config, manifest = _compaction_inputs()
    _, service = build_run_service(config=config)
    events = CollectedEvents(
        pause_on=lambda event: event["event"] == "context_compaction"
        and event["data"]["status"] == cancelled_at
    )
    run = asyncio.create_task(
        run_with_events(service, manifest, submit_operation=MagicMock(), events=events)
    )
    await events.wait_for(lambda event: event["data"].get("status") == cancelled_at)
    run.cancel()
    if cancelled_at == "started":
        # 还没有可交回的内容：取消原样传播，进程按无输出取消处理。
        with pytest.raises(asyncio.CancelledError):
            await run
        return
    result, _ = await run
    assert result.status == "cancelled"
    assert result.has_output is False
    assert result.compaction_checkpoint.reduced_through == "old"


def _summary_inputs():
    """使用需要接力摘要的多轮纯文本历史，保持生成请求在小预算内。"""
    config, manifest = _compaction_inputs()
    turns = tuple(
        make_conversation_turn(
            interaction_id=f"t{i}", user_message="x" * 400, assistant_final_text="y" * 400
        )
        for i in range(4)
    )
    return config, manifest.model_copy(update={"session_history": turns})


@pytest.mark.asyncio
async def test_cancellation_during_summary_propagates_without_partial_checkpoint():
    """减重已完成但摘要仍在等待时取消，不交回本次任何局部检查点；没有可交回的内容，取消原样传播。"""
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def summarize(_messages, _params):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    config, manifest = _summary_inputs()
    _, service = build_run_service(
        config=config, llm_service=CallbackLLMService(complete=summarize)
    )
    events = CollectedEvents()
    run = asyncio.create_task(
        run_with_events(service, manifest, submit_operation=MagicMock(), events=events)
    )
    await entered.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert [event["data"]["status"] for event in events.events] == ["started"]
    assert closed.is_set()


@pytest.mark.asyncio
async def test_abandoned_summary_still_generates_from_reduced_view_if_it_fits():
    """摘要被拒绝但减重视图可执行时继续生成，摘要失败不会变成本轮输出。"""

    async def empty_summary(_messages, _params):
        return LLMCompletion(text="", finish_reason="stop")

    async def generate(_messages, _params):
        yield "继续完成"

    llm = CallbackLLMService(complete=empty_summary, stream=generate)
    config, manifest = _summary_inputs()
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder, llm_service=llm)
    result, events = await run_with_events(service, manifest, submit_operation=MagicMock())
    assert events[1]["data"]["status"] == "abandoned"
    assert events[1]["data"]["layer"] == "summary"
    assert result.status == "completed"
    assert result.compaction_checkpoint.summary == ""
    assert [event.kind for event in result.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    (prompt,) = [call.messages for call in llm.calls if call.stream]
    assert all("Earlier conversation summary" not in message["content"] for message in prompt)
    assert [
        event.reason
        for event in recorder.events
        if event.event_type == RuntimeEventType.AGENT_CONTEXT_COMPACTION
    ] == [None, "summary_empty"]
