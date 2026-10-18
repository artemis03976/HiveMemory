"""
AgentRunService 集成测试 — 真实 AgentRuntime 装配链协作

驱动 AgentRunService、内核工厂装配的 AgentRuntime、CALL 编排、帧工厂、
提示词组装与真实事件管线；模型输出通过执行端口或 LLM 客户端替身提供。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.model.client import LLMClient
from hivememory.agent_runtime.output import TokenDelta
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.run.output import NullRunOutput
from hivememory.components.events.bus import (
    RecordingRuntimeEventSink,
)
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.workspace.contracts import CPUExecutionStatus
from tests.helpers.alice import build_run_service, stream_run
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.sessions import make_conversation_turn
from tests.helpers.workspace import make_execution_labels


@pytest.mark.asyncio
async def test_session_model_and_output_override_control_budget_and_provider_request(
    tmp_path, monkeypatch
):
    """会话覆盖模型与输出上限后，预检预算和实际模型请求使用同一次解析。"""
    from hivememory.agent_runtime.model.client import GenerationResult, LLMClient
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
    _, service = build_run_service(config=config, model_registry=registry)
    profile = AgentProfile(model_name="unavailable-profile-model", persona="", temperature=0.2)
    manifest = make_input_manifest(agent_profile=profile, user_message="x" * 2000)
    requests = []

    async def generate(_self, messages, **kwargs):
        requests.append(kwargs)
        return GenerationResult(text="完成")

    monkeypatch.setattr(LLMClient, "generate_async", generate)
    rejected = await service.run_agent(
        manifest, {"model": "small"}, run_output=NullRunOutput(), submit_operation=MagicMock()
    )
    assert rejected.failure_reason == "input_too_large"
    assert requests == []

    completed = await service.run_agent(
        manifest,
        {"model": "small", "max_tokens": 100},
        run_output=NullRunOutput(),
        submit_operation=MagicMock(),
    )
    assert completed.status == "completed"
    assert completed.final_text == "完成"
    assert completed.model_used == "小窗口"
    assert requests == [
        {
            "model": "provider/small",
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 1.0,
            "api_key": None,
            "api_base": None,
        }
    ]


@pytest.mark.asyncio
async def test_call_child_guard_uses_callee_model_and_returns_error_to_parent(
    tmp_path, monkeypatch
):
    """子帧按自己的小窗口被拦截，父帧收到 CALL 错误后继续完成。"""
    from hivememory.agent_runtime.model.client import GenerationResult, LLMClient
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
    _, service = build_run_service(config=config, model_registry=registry, runtime_events=recorder)
    requests = []

    async def generate(_self, messages, **kwargs):
        requests.append(kwargs["model"])
        if len(requests) == 1:
            text = '⟪ CALL | helper | task="总结" ⟫'
            return GenerationResult(text=text, was_mtp_interrupted=True, mtp_fragment=text)
        return GenerationResult(text="父帧完成")

    async def derive(_request):
        return make_callee(AgentProfile(model_name="child", persona="x" * 1000))

    monkeypatch.setattr(LLMClient, "generate_async", generate)
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
    assert requests == ["provider/parent", "provider/parent"]
    (guard,) = [e for e in recorder.events if e.event_type == "agent.context.budget_exceeded"]
    assert guard.data["input_token_limit"] == 100
    assert recorder.events[-1].event_type == "agent.run.completed"


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id="agent-1",
            user_id="u1",
            confidence_score=0.9,
        ),
        index=IndexLayer(
            title="test memory",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.FACT,
            alias="mem_alias",
        ),
        payload=PayloadLayer(content="memory content"),
    )


def _build_input_manifest(
    memory: MemoryAtom,
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
        memories=[memory],
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
    memory = _build_memory_atom()
    context = _build_input_manifest(memory).model_copy(
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


async def _terminal_status_without_stream(service, manifest, submit_operation) -> str:
    """非流式入口直接返回终态结果。"""
    result = await service.run_agent(
        manifest, run_output=NullRunOutput(), submit_operation=submit_operation
    )
    return result.status


async def _terminal_status_from_stream(service, manifest, submit_operation) -> str:
    """流式入口以最后一个事件携带终态。"""
    events = [
        event async for event in stream_run(service, manifest, submit_operation=submit_operation)
    ]
    return events[-1].status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_to_terminal",
    [
        pytest.param(_terminal_status_without_stream, id="non-stream"),
        pytest.param(_terminal_status_from_stream, id="stream"),
    ],
)
async def test_root_policy_and_prompt_follow_manifest_operations(run_to_terminal):
    """根 frame 的策略与教学由清单的操作快照翻译，系统工具菜单不受操作集合裁剪。"""
    runtime, service = build_run_service()
    manifest = _build_input_manifest(
        _build_memory_atom(),
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
    assert await run_to_terminal(service, manifest, submit_operation) == "completed"

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
    monkeypatch, profile_agent_id, display_agent_id
) -> None:
    """真实执行链不携带授权身份，事件用注册标签，流展示优先 Profile 源 alias。"""
    from types import SimpleNamespace

    from hivememory.core.models import AgentProfile

    recorder = RecordingRuntimeEventSink()
    _runtime, service = build_run_service(runtime_events=recorder)
    manifest = _build_input_manifest(
        _build_memory_atom(),
        labels=make_execution_labels(
            agent_id="registered_agent", workspace_id="isolation_workspace"
        ),
        agent_profile=AgentProfile(agent_id=profile_agent_id),
        process_id="interaction-isolation",
    )

    async def completion(**_kwargs):
        async def chunks():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="标签保持一致"), finish_reason="stop"
                    )
                ]
            )

        return chunks()

    monkeypatch.setattr("hivememory.agent_runtime.model.client.litellm.acompletion", completion)
    events = [
        event
        async for event in stream_run(
            service,
            manifest,
            {"model": "test-model"},
            submit_operation=MagicMock(),
        )
    ]

    assert [event["event"] for event in events[:-1]] == ["token"]
    assert [event["data"]["agent_id"] for event in events[:-1]] == [display_agent_id]
    assert events[-1].final_text == "标签保持一致"
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
    context = _build_input_manifest(_build_memory_atom(), process_id="process-1")
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
    context = _build_input_manifest(_build_memory_atom())
    _stub_terminal_execution(runtime, FrameExecutionStatus.FAILED)

    result = await service.run_agent(
        context, run_output=NullRunOutput(), submit_operation=MagicMock()
    )

    assert result.status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].severity == "error"


@pytest.mark.asyncio
async def test_run_agent_stream_returns_completed_root_frame():
    runtime, service = build_run_service()
    memory = _build_memory_atom()
    context = _build_input_manifest(memory)
    _stub_terminal_execution(runtime)

    events = [event async for event in stream_run(service, context, submit_operation=MagicMock())]

    assert [event["event"] for event in events[:-1]] == []
    assert events[0].status == CPUExecutionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())

    async def _run_frame(_frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="hi"))
        await asyncio.Event().wait()

    runtime.run_frame = _run_frame
    stream = stream_run(service, context, submit_operation=MagicMock())

    assert (await anext(stream))["event"] == "token"
    await stream.aclose()

    runtime_event_types = [event.event_type for event in recorder.events]
    assert RuntimeEventType.AGENT_RUN_STARTED in runtime_event_types
    assert RuntimeEventType.AGENT_RUN_CANCELLED in runtime_event_types
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"


@pytest.mark.asyncio
async def test_run_agent_stream_error_preserves_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())
    runtime.run_frame = AsyncMock(side_effect=RuntimeError("network unavailable"))

    with pytest.raises(RuntimeError, match="network unavailable"):
        async for _ in stream_run(service, context, submit_operation=MagicMock()):
            pass

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"


@pytest.mark.asyncio
async def test_unified_entry_stream_and_once_agree_on_terminal_outcome():
    """同一段脚本化运行：流式最后的类型化结果与非流式结果在终态、回复、轮次事件与
    模型名上一致；两种模式都发布 started 与终态事件，终态载荷仍含两个统计字段。"""
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

    stream_recorder = RecordingRuntimeEventSink()
    stream_runtime, stream_service = build_run_service(runtime_events=stream_recorder)
    stream_runtime.run_frame = _scripted_run_frame

    stream_events = [
        event
        async for event in stream_run(
            stream_service,
            _build_input_manifest(_build_memory_atom(), process_id="process-unified"),
            submit_operation=MagicMock(),
        )
    ]
    done = stream_events[-1]

    once_recorder = RecordingRuntimeEventSink()
    once_runtime, once_service = build_run_service(runtime_events=once_recorder)
    once_runtime.run_frame = _scripted_run_frame

    result = await once_service.run_agent(
        _build_input_manifest(_build_memory_atom(), process_id="process-unified"),
        run_output=NullRunOutput(),
        submit_operation=MagicMock(),
    )

    # 两种模式在终态、最终回复、轮次事件与模型名上一致。
    assert result.status == done.status == CPUExecutionStatus.COMPLETED.value
    assert result.final_text == done.final_text == "hello world"
    assert [event.kind for event in result.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    assert done.turn_events == result.turn_events
    assert result.model_used == done.model_used == "glm-4"
    # Alice 的迭代统计不进入执行结果，只保留在 agent.run.* 观测载荷中。
    for recorder in (stream_recorder, once_recorder):
        types = [event.event_type for event in recorder.events]
        assert RuntimeEventType.AGENT_RUN_STARTED in types
        terminal = recorder.events[-1]
        assert terminal.event_type == RuntimeEventType.AGENT_RUN_COMPLETED
        assert terminal.data["mtp_iterations"] == 2
        assert terminal.data["total_iterations"] == 3


@pytest.mark.asyncio
async def test_non_streaming_cancellation_publishes_cancelled_runtime_event():
    """统一入口后，非流式被取消同样发布 agent.run.cancelled（观测语义与流式一致）。"""
    recorder = RecordingRuntimeEventSink()
    runtime, service = build_run_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())
    started = asyncio.Event()

    async def _run_frame(_frame, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    runtime.run_frame = _run_frame

    task = asyncio.create_task(
        service.run_agent(context, run_output=NullRunOutput(), submit_operation=MagicMock())
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_CANCELLED
    assert recorder.events[-1].status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("during_pull", [True, False])
async def test_native_cancellation_fills_slot_with_delivered_partial_text(monkeypatch, during_pull):
    """真实 AgentLoop/RunExecutor 收口半截文字；关闭与取消都在异常传播前回填结果槽。"""
    from hivememory.agent_runtime.model.client import LLMClient, StreamChunk
    from hivememory.workspace.contracts import CPUResultSlot

    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def generate_stream(self, *_args, **_kwargs):
        try:
            yield StreamChunk(delta="半截答案", full_text="半截答案")
            entered.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(LLMClient, "generate_stream", generate_stream)
    from hivememory.agent_runtime.model.client import GenerationResult

    monkeypatch.setattr(
        LLMClient,
        "generate_async",
        AsyncMock(return_value=GenerationResult(text="Accepted summary")),
    )
    config, manifest = _summary_inputs()
    manifest = manifest.model_copy(update={"user_message": "hello"})
    _runtime, service = build_run_service(config=config)
    slot = CPUResultSlot()
    stream = stream_run(
        service,
        manifest,
        submit_operation=MagicMock(),
        result_slot=slot,
    )
    assert (await anext(stream))["data"]["status"] == "started"
    assert (await anext(stream))["data"]["status"] == "completed"
    assert (await anext(stream))["data"]["content"] == "半截答案"
    await entered.wait()
    if during_pull:
        pull = asyncio.create_task(anext(stream))
        # 下一次事件拉取开始后注入原生取消；不依赖固定 sleep。
        asyncio.get_running_loop().call_soon(pull.cancel)
        with pytest.raises(asyncio.CancelledError):
            await pull
    else:
        await stream.aclose()
    assert cancelled.is_set()
    assert slot.result.status == "cancelled"
    assert slot.result.final_text == "半截答案"
    assert slot.result.compaction_checkpoint.summary == "Accepted summary"
    assert slot.result.compaction_checkpoint.summarized_through == "t2"
    assert [(e.kind, e.content, e.status) for e in slot.result.turn_events] == [
        ("user_message", "hello", None),
        ("assistant_message", "半截答案", "cancelled"),
    ]


@pytest.mark.asyncio
async def test_cancel_in_call_keeps_root_call_without_child_partial_text(monkeypatch):
    """子帧中断时只保存根帧的不完整 CALL，不把尚未回填的子回复写成根回复。"""
    from hivememory.agent_runtime.model.client import GenerationResult, LLMClient, StreamChunk
    from hivememory.workspace.contracts import CPUResultSlot
    from tests.helpers.call import StubOperationSubmitter, make_callee

    calls = 0
    child_entered = asyncio.Event()

    async def generate_stream(self, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            command = '⟪ CALL | helper | task="分析" ⟫'
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text=command,
                    prefix_text="",
                    mtp_fragment=command,
                    was_mtp_interrupted=True,
                ),
            )
        else:
            yield StreamChunk(delta="子帧半截", full_text="子帧半截")
            child_entered.set()
            await asyncio.Event().wait()

    async def derive(_request):
        return make_callee(AgentProfile(agent_id="helper"))

    monkeypatch.setattr(LLMClient, "generate_stream", generate_stream)
    _runtime, service = build_run_service()
    slot = CPUResultSlot()
    stream = stream_run(
        service,
        _build_input_manifest(_build_memory_atom()),
        submit_operation=StubOperationSubmitter(derive=derive),
        result_slot=slot,
    )
    while (await anext(stream))["event"] != "token":
        pass
    await child_entered.wait()
    await stream.aclose()
    assert slot.result.status == "cancelled"
    assert slot.result.final_text == ""
    assert [(e.kind, e.tool_kind) for e in slot.result.turn_events] == [
        ("user_message", None),
        ("tool_call", "CALL"),
    ]
    assert slot.result.has_output is True


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
@pytest.mark.parametrize("stream", [False, True])
async def test_compaction_reaches_generation_and_returns_checkpoint_without_recording_progress(
    monkeypatch, stream
):
    """减重不调用辅助模型；正常生成消费缩小后的视图，进度不进入交互载荷。"""
    from hivememory.agent_runtime.model.client import GenerationResult, LLMClient, StreamChunk
    from hivememory.workspace.contracts import CPUResultSlot

    requests = []

    async def generate(_self, messages, **_kwargs):
        requests.append(messages)
        return GenerationResult(text="完成")

    async def generate_stream(_self, messages, **_kwargs):
        yield StreamChunk(is_final=True, result=await generate(_self, messages))

    monkeypatch.setattr(LLMClient, "generate_async", generate)
    monkeypatch.setattr(LLMClient, "generate_stream", generate_stream)
    config, manifest = _compaction_inputs()
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder)
    slot = CPUResultSlot()
    if stream:
        events = [
            event
            async for event in stream_run(
                service, manifest, submit_operation=MagicMock(), result_slot=slot
            )
        ]
        progress = [
            event
            for event in events
            if isinstance(event, dict) and event["event"] == "context_compaction"
        ]
        assert [event["data"]["status"] for event in progress] == ["started", "completed"]
        assert [event["data"]["stream_sequence"] for event in events[:-1]] == list(
            range(len(events) - 1)
        )
        assert events[-1] is slot.result
        result = slot.result
    else:
        result = await service.run_agent(
            manifest,
            run_output=NullRunOutput(),
            submit_operation=MagicMock(),
            result_slot=slot,
        )
    assert result.status == "completed"
    assert result.compaction_checkpoint.reduced_through == "old"
    assert [(event.kind, event.content) for event in result.turn_events] == [
        ("user_message", "next"),
        ("assistant_message", "完成"),
    ]
    assert len(requests) == 1
    assert "READ again" in requests[0][1]["content"]
    assert all("x" * 6000 not in message["content"] for message in requests[0])
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
async def test_uncompressible_history_fails_without_output_and_keeps_reduction_checkpoint(
    monkeypatch,
):
    """用户原文不能减重，仍超限时不调用生成模型；进度本身不构成输出。"""
    from hivememory.agent_runtime.model.client import LLMClient
    from hivememory.workspace.contracts import CPUResultSlot

    generate = AsyncMock(side_effect=AssertionError("超限不应调用模型"))
    monkeypatch.setattr(LLMClient, "generate_async", generate)
    config, manifest = _compaction_inputs(compressible=False)
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder)
    slot = CPUResultSlot()
    events = [
        event
        async for event in stream_run(
            service, manifest, submit_operation=MagicMock(), result_slot=slot
        )
    ]
    assert [event["event"] for event in events[:-1]] == [
        "context_compaction",
        "context_compaction",
    ]
    assert events[1]["data"]["status"] == "failed"
    assert slot.result.failure_reason == "compaction_failed"
    assert slot.result.has_output is False
    assert slot.result.compaction_checkpoint.reduced_through == "old"
    generate.assert_not_awaited()
    failed_compaction = [
        event
        for event in recorder.events
        if event.event_type == RuntimeEventType.AGENT_CONTEXT_COMPACTION
        and event.status == "failed"
    ]
    assert [event.reason for event in failed_compaction] == ["summary_input_too_large"]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_completion", [False, True])
async def test_closing_at_compaction_progress_only_returns_completed_checkpoint(after_completion):
    """压缩开始时关闭不产生新检查点，结束事件交出后关闭则交回已完成视图。"""
    from hivememory.workspace.contracts import CPUResultSlot

    config, manifest = _compaction_inputs()
    _, service = build_run_service(config=config)
    slot = CPUResultSlot()
    stream = stream_run(service, manifest, submit_operation=MagicMock(), result_slot=slot)
    assert (await anext(stream))["data"]["status"] == "started"
    if after_completion:
        assert (await anext(stream))["data"]["status"] == "completed"
    await stream.aclose()
    if after_completion:
        assert slot.result.status == "cancelled"
        assert slot.result.compaction_checkpoint.reduced_through == "old"
        assert slot.result.has_output is False
    else:
        assert slot.result is None


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
async def test_cancellation_during_summary_propagates_without_new_checkpoint(monkeypatch):
    """减重已完成但摘要仍在等待时取消，不能交回本次任何局部检查点。"""
    from hivememory.workspace.contracts import CPUResultSlot

    entered = asyncio.Event()
    closed = asyncio.Event()

    async def summarize(_self, messages, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(LLMClient, "generate_async", summarize)
    config, manifest = _summary_inputs()
    _, service = build_run_service(config=config)
    slot = CPUResultSlot()
    stream = stream_run(service, manifest, submit_operation=MagicMock(), result_slot=slot)
    assert (await anext(stream))["data"]["status"] == "started"
    pull = asyncio.create_task(anext(stream))
    await entered.wait()
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    await stream.aclose()
    assert closed.is_set()
    assert slot.result is None


@pytest.mark.asyncio
async def test_abandoned_summary_still_generates_from_reduced_view_if_it_fits(monkeypatch):
    """摘要被拒绝但减重视图可执行时继续生成，摘要失败不会变成本轮输出。"""
    from hivememory.agent_runtime.model.client import GenerationResult, StreamChunk
    from hivememory.workspace.contracts import CPUResultSlot

    monkeypatch.setattr(
        LLMClient,
        "generate_async",
        AsyncMock(return_value=GenerationResult(text="", finish_reason="stop")),
    )
    prompts = []

    async def generate(_self, messages, **kwargs):
        prompts.append(messages)
        yield StreamChunk(is_final=True, result=GenerationResult(text="继续完成"))

    monkeypatch.setattr(LLMClient, "generate_stream", generate)
    config, manifest = _summary_inputs()
    recorder = RecordingRuntimeEventSink()
    _, service = build_run_service(config=config, runtime_events=recorder)
    slot = CPUResultSlot()
    events = [
        event
        async for event in stream_run(
            service, manifest, submit_operation=MagicMock(), result_slot=slot
        )
    ]
    assert events[1]["data"]["status"] == "abandoned"
    assert events[1]["data"]["layer"] == "summary"
    assert slot.result.status == "completed"
    assert slot.result.compaction_checkpoint.summary == ""
    assert [event.kind for event in slot.result.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    assert len(prompts) == 1
    assert all("Earlier conversation summary" not in message["content"] for message in prompts[0])
    assert [
        event.reason
        for event in recorder.events
        if event.event_type == RuntimeEventType.AGENT_CONTEXT_COMPACTION
    ] == [None, "summary_empty"]
