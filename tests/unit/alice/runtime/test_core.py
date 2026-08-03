import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.protocol.models import AgentRunContext, AgentRunStatus, RetrievalResponse
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(
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
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        topic_context=None,
        retrieval_result=RetrievalResponse(memories=[memory]),
        memory_context="ctx",
        agent_profile=OMNI_DOLL_PROFILE,
        storage_available=True,
    )


def _build_runtime(*, runtime_events=None) -> AliceRuntime:
    config = HiveMemoryConfig()
    return AliceRuntime(
        alice_config=config.alice,
        shared_config=config.shared,
        memory_compiler_config=config.memory_compiler,
        runtime_events=runtime_events,
    )


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
    runtime = _build_runtime()
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    _stub_terminal_execution(runtime)

    await runtime.run_agent(context)

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    runtime._agent_runtime.run_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_correlates_runtime_scope_and_generation_id():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    _stub_terminal_execution(runtime)
    created_sessions = []
    create_run_session = runtime._create_run_session

    def _capture_session(**kwargs):
        session = create_run_session(**kwargs)
        created_sessions.append(session)
        return session

    runtime._create_run_session = _capture_session
    cancel_event = asyncio.Event()

    await runtime.run_agent(
        context,
        cancel_event=cancel_event,
        generation_id="generation-1",
    )

    session = created_sessions[0]
    assert session.generation_id == "generation-1"
    assert session.agent_run_id == recorder.events[0].agent_run_id
    assert session.cancel_event is cancel_event
    assert runtime._agent_runtime.run_frame.await_args.kwargs["cancel_event"] is cancel_event


@pytest.mark.asyncio
async def test_run_agent_failed_result_emits_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    _stub_terminal_execution(runtime, FrameExecutionStatus.FAILED)

    result = await runtime.run_agent(context)

    assert result.status == AgentRunStatus.FAILED.value
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == AgentRunStatus.FAILED.value
    assert recorder.events[-1].severity == "error"


@pytest.mark.asyncio
async def test_run_agent_stream_warms_preretrieval_alias_cache_before_execution():
    runtime = _build_runtime()
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    _stub_terminal_execution(runtime)

    events = [event async for event in runtime.run_agent_stream(context)]

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    assert [event["event"] for event in events] == ["done"]
    assert events[0]["data"]["status"] == AgentRunStatus.COMPLETED.value
    assert events[0]["data"]["scope"] == "main"


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    received_cancel_events = []

    async def _run_frame(_frame, *, event_sink, cancel_event, **_kwargs):
        received_cancel_events.append(cancel_event)
        await event_sink.emit({"event": "token", "data": {"content": "hi"}})
        await asyncio.Event().wait()

    runtime._agent_runtime.run_frame = _run_frame
    runtime._agent_runtime.finalize_run = MagicMock(return_value=RuntimeProducts())
    stream = runtime.run_agent_stream(context)

    assert (await anext(stream))["event"] == "token"
    await stream.aclose()

    runtime_event_types = [event.event_type for event in recorder.events]
    assert RuntimeEventType.AGENT_RUN_STARTED in runtime_event_types
    assert RuntimeEventType.AGENT_RUN_CANCELLED in runtime_event_types
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"
    assert len(received_cancel_events) == 1
    assert received_cancel_events[0].is_set()
    runtime._agent_runtime.finalize_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_stream_error_does_not_set_cancel_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    cancel_event = asyncio.Event()
    runtime._agent_runtime.run_frame = AsyncMock(side_effect=RuntimeError("network unavailable"))

    with pytest.raises(RuntimeError, match="network unavailable"):
        async for _ in runtime.run_agent_stream(context, cancel_event=cancel_event):
            pass

    assert cancel_event.is_set() is False
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"


@pytest.mark.asyncio
async def test_run_agent_stream_without_scheduler_terminal_fails_cleanly():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    cancel_event = asyncio.Event()

    async def _stream(*_args, **_kwargs):
        yield {"event": "token", "data": {"content": "hi"}}

    scheduler = MagicMock()
    scheduler.run_stream = _stream
    scheduler.terminal_result = None
    with patch("hivememory.alice.runtime.core.RunScheduler", return_value=scheduler):
        with pytest.raises(RuntimeError, match="ended without done"):
            async for _ in runtime.run_agent_stream(context, cancel_event=cancel_event):
                pass

    assert cancel_event.is_set() is False
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].message == "Agent stream ended without done event."
