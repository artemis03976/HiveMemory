import asyncio
from unittest.mock import AsyncMock

import pytest

from hivememory.alice.runtime.core import AliceRuntime
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
)
from hivememory.core.protocol.models import AgentRunContext, AgentRunResult, RetrievalResponse
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.config import HiveMemoryConfig, AliceConfig, SharedConfig
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
        retrieval_result=RetrievalResponse(memories=[memory], rendered_context="ctx"),
        agent_profile=OMNI_DOLL_PROFILE,
        storage_available=True,
    )


@pytest.mark.asyncio
async def test_run_agent_warms_preretrieval_alias_cache_before_execution():
    runtime = AliceRuntime(alice_config=HiveMemoryConfig().alice, shared_config=HiveMemoryConfig().shared)
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
async def test_run_agent_stream_warms_preretrieval_alias_cache_before_execution():
    runtime = AliceRuntime(alice_config=HiveMemoryConfig().alice, shared_config=HiveMemoryConfig().shared)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)

    async def _stream(**kwargs):
        yield {"event": "done", "data": AgentRunResult(final_text="done").model_dump()}

    runtime._orchestrator.run_agent_stream = _stream

    events = [event async for event in runtime.run_agent_stream(context)]

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    assert events == [
        {"event": "done", "data": AgentRunResult(final_text="done").model_dump()}
    ]


@pytest.mark.asyncio
async def test_run_agent_stream_close_emits_cancelled_runtime_event():
    recorder = RecordingRuntimeEventSink()
    runtime = AliceRuntime(alice_config=HiveMemoryConfig().alice, shared_config=HiveMemoryConfig().shared, runtime_events=recorder)
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)

    async def _stream(**kwargs):
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


@pytest.mark.asyncio
async def test_run_agent_stream_error_does_not_set_cancel_event():
    recorder = RecordingRuntimeEventSink()
    runtime = AliceRuntime(alice_config=HiveMemoryConfig().alice, shared_config=HiveMemoryConfig().shared, runtime_events=recorder)
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
    runtime = AliceRuntime(alice_config=HiveMemoryConfig().alice, shared_config=HiveMemoryConfig().shared, runtime_events=recorder)
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
