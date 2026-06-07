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
from hivememory.system.config import HiveMemoryConfig


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
        topic_context={"blocks": [], "state_summary": ""},
        retrieval_result=RetrievalResponse(memories=[memory], rendered_context="ctx"),
        agent_profile=OMNI_DOLL_PROFILE,
        storage_available=True,
    )


@pytest.mark.asyncio
async def test_run_agent_warms_preretrieval_alias_cache_before_execution():
    runtime = AliceRuntime(config=HiveMemoryConfig())
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
    runtime = AliceRuntime(config=HiveMemoryConfig())
    memory = _build_memory_atom()
    context = _build_agent_run_context(memory)

    async def _stream(**kwargs):
        yield {"event": "done"}

    runtime._orchestrator.run_agent_stream = _stream

    events = [event async for event in runtime.run_agent_stream(context)]

    cached = runtime._koakuma.atom_cache.get_atom_by_alias("mem_alias")
    assert cached is memory
    assert events == [{"event": "done"}]
