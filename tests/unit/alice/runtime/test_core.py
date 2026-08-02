import asyncio
from unittest.mock import AsyncMock

import pytest

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
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AgentRunStatus,
    RetrievalResponse,
)
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


@pytest.mark.asyncio
async def test_run_agent_warms_preretrieval_alias_cache_before_execution():
    runtime = _build_runtime()
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    runtime._orchestrator.run_agent = AsyncMock(
        return_value=AgentRunResult(final_text="done"),
    )

    await runtime.run_agent(context)

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    runtime._orchestrator.run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_correlates_runtime_scope_and_generation_id():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    runtime._orchestrator.run_agent = AsyncMock(
        return_value=AgentRunResult(final_text="done"),
    )

    cancel_event = asyncio.Event()
    await runtime.run_agent(
        context,
        cancel_event=cancel_event,
        generation_id="generation-1",
    )

    kwargs = runtime._orchestrator.run_agent.await_args.kwargs
    session = kwargs["session"]
    assert session.generation_id == "generation-1"
    assert session.agent_run_id == recorder.events[0].agent_run_id
    assert session.cancel_event is cancel_event
    assert "generation_id" not in kwargs
    assert "agent_run_id" not in kwargs
    assert "cancel_event" not in kwargs


@pytest.mark.asyncio
async def test_run_agent_failed_result_emits_failed_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    runtime._orchestrator.run_agent = AsyncMock(
        return_value=AgentRunResult(status=AgentRunStatus.FAILED),
    )

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

    async def _stream(**kwargs):
        yield {"event": "done", "data": AgentRunResult(final_text="done").model_dump()}

    runtime._orchestrator.run_agent_stream = _stream

    events = [event async for event in runtime.run_agent_stream(context)]

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    assert events == [{"event": "done", "data": AgentRunResult(final_text="done").model_dump()}]


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)

    received_sessions = []

    async def _stream(**kwargs):
        received_sessions.append(kwargs["session"])
        yield {"event": "token", "data": {"content": "hi"}}
        await asyncio.Event().wait()

    runtime._orchestrator.run_agent_stream = _stream

    stream = runtime.run_agent_stream(context)
    assert (await stream.__anext__())["event"] == "token"

    await stream.aclose()

    runtime_event_types = [event.event_type for event in recorder.events]
    assert RuntimeEventType.AGENT_RUN_STARTED in runtime_event_types
    assert RuntimeEventType.AGENT_RUN_CANCELLED in runtime_event_types
    assert recorder.events[-1].status == "cancelled"
    assert recorder.events[-1].data["close_reason"] == "stream_closed"
    assert len(received_sessions) == 1
    assert received_sessions[0].cancel_event.is_set()


@pytest.mark.asyncio
async def test_run_agent_stream_error_does_not_set_cancel_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    cancel_event = asyncio.Event()

    async def _stream(**kwargs):
        raise RuntimeError("network unavailable")
        yield

    runtime._orchestrator.run_agent_stream = _stream

    with pytest.raises(RuntimeError, match="network unavailable"):
        async for _ in runtime.run_agent_stream(context, cancel_event=cancel_event):
            pass

    assert cancel_event.is_set() is False
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].status == "failed"


@pytest.mark.asyncio
async def test_run_agent_stream_without_done_fails_without_setting_cancel_event():
    recorder = RecordingRuntimeEventSink()
    runtime = _build_runtime(runtime_events=recorder)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)
    cancel_event = asyncio.Event()

    async def _stream(**kwargs):
        yield {"event": "token", "data": {"content": "hi"}}

    runtime._orchestrator.run_agent_stream = _stream

    with pytest.raises(RuntimeError, match="ended without done"):
        async for _ in runtime.run_agent_stream(context, cancel_event=cancel_event):
            pass

    assert cancel_event.is_set() is False
    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].message == "Agent stream ended without done event."
