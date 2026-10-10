"""
AgentRunService 集成测试 — 真实 AliceRuntime 装配链协作

驱动 AgentRunService + 真实 AliceRuntime（AgentRuntime 实例）+ 真实 CallCoordinator/CallContextProvider/FrameFactory/
AgentPromptAssembler + 真实事件管线；仅 stub LLM 执行端口 run_frame。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
)
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.contracts import CPUExecutionStatus
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


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
    identity_scope=None,
    process_id: str = "process-test",
):
    return make_input_manifest(
        identity_scope=identity_scope or make_identity_scope(user_id="u1", agent_id="omni_doll"),
        process_id=process_id,
        topic_id="topic_1",
        user_message="hello",
        memories=[memory],
        memory_context="ctx",
    )


def _build_service(*, runtime_events=None) -> tuple[AliceRuntime, AgentRunService]:
    config = HiveMemoryConfig()
    runtime = AliceRuntime(
        alice_config=config.alice,
        memory_compiler_config=config.memory_compiler,
    )
    frame_factory = FrameFactory()
    prompt_assembler = AgentPromptAssembler(config.alice.koakuma)
    coordinator = CallCoordinator(
        runtime.agent_runtime,
        CallContextProvider(),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )
    service = AgentRunService(
        agent_runtime=runtime.agent_runtime,
        call_coordinator=coordinator,
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
        stream_adapter=AgentRunStreamAdapter(),
        agent_run_events=AgentRunEventEmitter(
            RuntimeEventPublisher(runtime_events or NullRuntimeEventSink())
        ),
    )
    return runtime, service


def _stub_terminal_execution(
    runtime: AliceRuntime,
    status: FrameExecutionStatus = FrameExecutionStatus.COMPLETED,
) -> None:
    runtime._agent_runtime.run_frame = AsyncMock(
        return_value=FrameExecutionResult(status=status),
    )


@pytest.mark.asyncio
async def test_run_agent_passes_input_and_submitter_to_root_frame():
    """CPU 交付的提交函数随 root frame 进入执行层，预检索内容仍进入提示词。"""
    runtime, service = _build_service()
    memory = _build_memory_atom()
    context = _build_input_manifest(memory)
    seen = []
    submit_operation = MagicMock()

    async def run_frame(frame, **_kwargs):
        seen.append(frame)
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime._agent_runtime.run_frame = run_frame
    result = await service.run_agent(context, stream=False, submit_operation=submit_operation)
    assert result.status == CPUExecutionStatus.COMPLETED.value
    (frame,) = seen
    assert frame.submit_operation is submit_operation
    assert any("hello" in message["content"] for message in frame.working_history)


@pytest.mark.asyncio
async def test_root_frame_inherits_agent_run_workspace_context() -> None:
    """防止 Alice 创建 root frame 时从 actor 字段重新拼装默认 Workspace。"""
    runtime, service = _build_service()
    context = _build_input_manifest(
        _build_memory_atom(),
        identity_scope=make_identity_scope(
            user_id="u1",
            agent_id="omni_doll",
            workspace_id="isolation_workspace",
        ),
        process_id="interaction-isolation",
    )
    _stub_terminal_execution(runtime)

    await service.run_agent(context, stream=False, submit_operation=MagicMock())

    frame = runtime._agent_runtime.run_frame.await_args.args[0]
    assert frame.identity_scope == context.identity_scope
    assert frame.runtime_scope.identity_scope == context.identity_scope


@pytest.mark.asyncio
async def test_run_agent_correlates_runtime_scope_and_process_id():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom(), process_id="process-1")
    _stub_terminal_execution(runtime)
    created_sessions = []
    create_run_session = service._create_run_session

    def _capture_session(**kwargs):
        session = create_run_session(**kwargs)
        created_sessions.append(session)
        return session

    service._create_run_session = _capture_session
    await service.run_agent(context, stream=False, submit_operation=MagicMock())

    session = created_sessions[0]
    assert session.process_id == "process-1"
    assert session.agent_run_id == recorder.events[0].agent_run_id
    assert recorder.events[0].process_id == "process-1"


@pytest.mark.asyncio
async def test_run_agent_failed_result_emits_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())
    _stub_terminal_execution(runtime, FrameExecutionStatus.FAILED)

    result = await service.run_agent(context, stream=False, submit_operation=MagicMock())

    assert result.status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == CPUExecutionStatus.FAILED.value
    assert recorder.events[-1].severity == "error"


@pytest.mark.asyncio
async def test_run_agent_stream_returns_completed_root_frame():
    runtime, service = _build_service()
    memory = _build_memory_atom()
    context = _build_input_manifest(memory)
    _stub_terminal_execution(runtime)

    events = [
        event
        async for event in service.run_agent(context, stream=True, submit_operation=MagicMock())
    ]

    assert [event["event"] for event in events] == ["done"]
    assert events[0]["data"]["status"] == CPUExecutionStatus.COMPLETED.value
    assert events[0]["data"]["scope"] == "main"


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())

    async def _run_frame(_frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="hi"))
        await asyncio.Event().wait()

    runtime._agent_runtime.run_frame = _run_frame
    stream = service.run_agent(context, stream=True, submit_operation=MagicMock())

    assert (await anext(stream))["event"] == "token"
    await stream.aclose()

    runtime_event_types = [event.event_type for event in recorder.events]
    assert RuntimeEventType.AGENT_RUN_STARTED in runtime_event_types
    assert RuntimeEventType.AGENT_RUN_CANCELLED in runtime_event_types
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"


@pytest.mark.asyncio
async def test_executor_stream_close_error_does_not_replace_task_cancellation():
    recorder = RecordingRuntimeEventSink()
    _runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())

    class CloseFailingExecutorStream:
        def __init__(self) -> None:
            self._emitted = False
            self.pull_started = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._emitted:
                self.pull_started.set()
                await asyncio.Event().wait()
            self._emitted = True
            return {"event": "token", "data": {"content": "hi"}}

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("executor stream close failed")

    executor_stream = CloseFailingExecutorStream()
    agent_stream = MagicMock()
    agent_stream.output = MagicMock()

    def events(runner):
        runner.close()
        return executor_stream

    agent_stream.events.side_effect = events
    service._stream_adapter.create = MagicMock(return_value=agent_stream)
    stream = service.run_agent(context, stream=True, submit_operation=MagicMock())

    assert (await anext(stream))["event"] == "token"
    pull_task = asyncio.create_task(anext(stream))
    await executor_stream.pull_started.wait()
    pull_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pull_task

    assert executor_stream.close_calls == 1
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_CANCELLED
    assert RuntimeEventType.AGENT_RUN_FAILED not in {event.event_type for event in recorder.events}


@pytest.mark.asyncio
async def test_run_agent_stream_error_preserves_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())
    runtime._agent_runtime.run_frame = AsyncMock(side_effect=RuntimeError("network unavailable"))

    with pytest.raises(RuntimeError, match="network unavailable"):
        async for _ in service.run_agent(context, stream=True, submit_operation=MagicMock()):
            pass

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"


@pytest.mark.asyncio
async def test_unified_entry_stream_and_once_agree_on_terminal_outcome():
    """同一段脚本化运行：流式最后的 done 与非流式结果在终态、回复、轮次事件与
    模型名上一致；两种模式都发布 started 与终态事件，终态载荷仍含两个统计字段。"""
    from hivememory.agent_runtime.models import FrameExecutionResult
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
    stream_runtime, stream_service = _build_service(runtime_events=stream_recorder)
    stream_runtime._agent_runtime.run_frame = _scripted_run_frame

    stream_events = [
        event
        async for event in stream_service.run_agent(
            _build_input_manifest(_build_memory_atom(), process_id="process-unified"),
            stream=True,
            submit_operation=MagicMock(),
        )
    ]
    done = next(event for event in stream_events if event["event"] == "done")

    once_recorder = RecordingRuntimeEventSink()
    once_runtime, once_service = _build_service(runtime_events=once_recorder)
    once_runtime._agent_runtime.run_frame = _scripted_run_frame

    result = await once_service.run_agent(
        _build_input_manifest(_build_memory_atom(), process_id="process-unified"),
        stream=False,
        submit_operation=MagicMock(),
    )

    # 两种模式在终态、最终回复、轮次事件与模型名上一致。
    assert result.status == done["data"]["status"] == CPUExecutionStatus.COMPLETED.value
    assert result.final_text == done["data"]["final_text"] == "hello world"
    assert [event.kind for event in result.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    assert done["data"]["turn_events"] == [event.model_dump() for event in result.turn_events]
    assert result.model_used == done["data"]["model_used"] == "glm-4"
    # Alice 的迭代统计不进入执行结果，只保留在 agent.run.* 观测载荷中。
    assert "mtp_iterations" not in done["data"]
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
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_input_manifest(_build_memory_atom())
    started = asyncio.Event()

    async def _run_frame(_frame, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    runtime._agent_runtime.run_frame = _run_frame

    task = asyncio.create_task(
        service.run_agent(context, stream=False, submit_operation=MagicMock())
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_CANCELLED
    assert recorder.events[-1].status == "cancelled"
