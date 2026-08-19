"""
AgentRunService 集成测试 — 真实 AliceRuntime 装配链协作

驱动 AgentRunService + 真实 AliceRuntime（profile/alias resolver、atom_cache、
AgentRuntime 实例）+ 真实 CallCoordinator/CallContextProvider/FrameFactory/
AgentPromptAssembler + 真实事件管线；仅 stub LLM 执行端口 run_frame/finalize_run。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.protocol.models import AgentRunContext, AgentRunStatus, RetrievalResponse
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from tests.helpers.workspace import make_access_context
from tests.helpers.memory import make_memory_metadata


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


def _build_agent_run_context(memory: MemoryAtom) -> AgentRunContext:
    return AgentRunContext(
        access_context=make_access_context(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        topic_context=None,
        retrieval_result=RetrievalResponse(memories=[memory]),
        memory_context="ctx",
        agent_profile=OMNI_DOLL_PROFILE,
        storage_available=True,
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
        CallContextProvider(runtime.profile_resolver, runtime.alias_resolver),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )
    service = AgentRunService(
        agent_runtime=runtime.agent_runtime,
        call_coordinator=coordinator,
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
        atom_cache=runtime.atom_cache,
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
    runtime._agent_runtime.finalize_run = MagicMock(return_value=RuntimeProducts())


@pytest.mark.asyncio
async def test_run_agent_warms_preretrieval_alias_cache_before_execution():
    runtime, service = _build_service()
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    _stub_terminal_execution(runtime)

    await service.run_agent(context)

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    runtime._agent_runtime.run_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_root_frame_inherits_agent_run_workspace_context() -> None:
    """防止 Alice 创建 root frame 时从 actor 字段重新拼装默认 Workspace。"""
    runtime, service = _build_service()
    context = _build_agent_run_context(_build_memory_atom()).model_copy(
        update={
            "access_context": make_access_context(
                user_id="u1",
                agent_id="omni_doll",
                workspace_id="isolation_workspace",
                interaction_id="interaction-isolation",
            )
        }
    )
    _stub_terminal_execution(runtime)

    await service.run_agent(context)

    frame = runtime._agent_runtime.run_frame.await_args.args[0]
    assert frame.access_context == context.access_context
    assert frame.runtime_scope.access_context.scope_fingerprint == (
        context.access_context.scope_fingerprint
    )


@pytest.mark.asyncio
async def test_run_agent_correlates_runtime_scope_and_generation_id():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    _stub_terminal_execution(runtime)
    created_sessions = []
    create_run_session = service._create_run_session

    def _capture_session(**kwargs):
        session = create_run_session(**kwargs)
        created_sessions.append(session)
        return session

    service._create_run_session = _capture_session
    await service.run_agent(
        context,
        generation_id="generation-1",
    )

    session = created_sessions[0]
    assert session.generation_id == "generation-1"
    assert session.agent_run_id == recorder.events[0].agent_run_id
    assert recorder.events[0].generation_id == "generation-1"


@pytest.mark.asyncio
async def test_run_agent_failed_result_emits_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    _stub_terminal_execution(runtime, FrameExecutionStatus.FAILED)

    result = await service.run_agent(context)

    assert result.status == AgentRunStatus.FAILED.value
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == AgentRunStatus.FAILED.value
    assert recorder.events[-1].severity == "error"


@pytest.mark.asyncio
async def test_run_agent_stream_warms_preretrieval_alias_cache_before_execution():
    runtime, service = _build_service()
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    _stub_terminal_execution(runtime)

    events = [event async for event in service.run_agent_stream(context)]

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    assert [event["event"] for event in events] == ["done"]
    assert events[0]["data"]["status"] == AgentRunStatus.COMPLETED.value
    assert events[0]["data"]["scope"] == "main"


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    async def _run_frame(_frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="hi"))
        await asyncio.Event().wait()

    runtime._agent_runtime.run_frame = _run_frame
    runtime._agent_runtime.finalize_run = MagicMock(return_value=RuntimeProducts())
    stream = service.run_agent_stream(context)

    assert (await anext(stream))["event"] == "token"
    await stream.aclose()

    runtime_event_types = [event.event_type for event in recorder.events]
    assert RuntimeEventType.AGENT_RUN_STARTED in runtime_event_types
    assert RuntimeEventType.AGENT_RUN_CANCELLED in runtime_event_types
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"
    runtime._agent_runtime.finalize_run.assert_called_once()


@pytest.mark.asyncio
async def test_executor_stream_close_error_does_not_replace_task_cancellation():
    recorder = RecordingRuntimeEventSink()
    _runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())

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
    stream = service.run_agent_stream(context)

    assert (await anext(stream))["event"] == "token"
    pull_task = asyncio.create_task(anext(stream))
    await executor_stream.pull_started.wait()
    pull_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pull_task

    assert executor_stream.close_calls == 1
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_CANCELLED
    assert RuntimeEventType.AGENT_RUN_FAILED not in {
        event.event_type for event in recorder.events
    }


@pytest.mark.asyncio
async def test_run_agent_stream_error_preserves_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    runtime._agent_runtime.run_frame = AsyncMock(side_effect=RuntimeError("network unavailable"))

    with pytest.raises(RuntimeError, match="network unavailable"):
        async for _ in service.run_agent_stream(context):
            pass

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"
