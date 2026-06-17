"""Patchouli prepare/finalize 通过 GlobalSystemBus 调用 Alice 的单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, OMNI_DOLL_PROFILE, TurnEvent
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.models import AgentRunContext, AgentRunResult, EyeGazeResult, RetrievalResponse
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(
            source_agent_id="agent-1",
            user_id="u1",
            confidence_score=0.9,
            access_count=1,
            vitality_score=88.0,
        ),
        index=IndexLayer(
            title="test memory",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.CODE_SNIPPET,
            alias="mem_alias",
        ),
        payload=PayloadLayer(content="print('hello')"),
    )


@pytest.mark.asyncio
async def test_prepare_agent_run_returns_agent_run_context_with_retrieval_result():
    kernel = MagicMock()
    eye = MagicMock()
    bus = GlobalSystemBus()
    local_bus = PatchouliBus()
    kernel.local_bus = local_bus
    kernel.check_storage_health = AsyncMock(return_value=True)

    gaze_result = EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="rewritten query",
        search_keywords=["tag"],
        worth_saving=True,
        raw_query="hi",
        identity=Identity(user_id="u1"),
        target_topic="topic_1",
    )
    retrieval_result = RetrievalResponse(
        memories=[_build_memory_atom()],
        rendered_context="<memory>ctx</memory>",
    )

    eye.gaze = AsyncMock(return_value=gaze_result)
    local_bus.register(
        "memory.get_agent_profile",
        AsyncMock(return_value=OMNI_DOLL_PROFILE),
    )
    local_bus.register(
        "librarian.get_active_topics_snapshots",
        AsyncMock(return_value=[]),
    )
    local_bus.register(
        "librarian.prepare_topic",
        AsyncMock(return_value=("topic_1", {"topics": []}, {"blocks": []})),
    )
    local_bus.register(
        "memory.retrieve",
        AsyncMock(
            return_value=retrieval_result
        ),
    )

    service = PatchouliService(runtime=kernel, eye=eye, global_bus=bus, local_bus=local_bus)

    prepared = await service.prepare_agent_run(
        user_message="hi",
        user_id="u1",
    )

    assert prepared.identity.user_id == "u1"
    assert prepared.topic_id == "topic_1"
    assert isinstance(prepared.agent_run_context, AgentRunContext)
    assert prepared.agent_run_context.retrieval_result is retrieval_result
    assert prepared.stream_prelude.memory_refs[0]["alias"] == "mem_alias"


@pytest.mark.asyncio
async def test_finalize_agent_run_reads_focus_from_loop_result():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.submit_interaction = AsyncMock(return_value=None)
    service = PatchouliService(
        runtime=kernel,
        eye=MagicMock(),
        global_bus=GlobalSystemBus(),
    )

    identity = Identity(user_id="u1", agent_id="omni_doll")
    prepared_run = PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context={"blocks": [], "state_summary": ""},
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="rewritten",
            search_keywords=[],
            worth_saving=True,
            raw_query="hi",
            identity=identity,
            target_topic="topic_1",
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
    )

    await service.finalize_agent_run(
        prepared_run=prepared_run,
        loop_result=AgentRunResult(
            final_text="done",
            turn_events=[
                TurnEvent(
                    kind="tool_call",
                    sequence=0,
                    role="assistant",
                    content="searching",
                    action_id="a1",
                    tool_kind="SEARCH",
                    tool_name="*",
                    tool_args={"query": "rewritten"},
                ),
                TurnEvent(
                    kind="tool_result",
                    sequence=1,
                    role="user",
                    content="result",
                    action_id="a1",
                    tool_kind="SEARCH",
                    tool_name="*",
                    status="success",
                ),
            ],
        ),
    )

    kernel.librarian_core.submit_interaction.assert_awaited_once()
    payload = kernel.librarian_core.submit_interaction.await_args.args[0]
    assert payload.mtp_traces
    assert payload.mtp_traces[0].action == "SEARCH"
    assert payload.materialize_tasks == []


@pytest.mark.asyncio
async def test_finalize_agent_run_records_retrieval_hits_once_per_memory():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.submit_interaction = AsyncMock(return_value=None)
    kernel.librarian_core.lifecycle_engine = MagicMock()
    service = PatchouliService(runtime=kernel, eye=MagicMock(), global_bus=GlobalSystemBus())

    memory = _build_memory_atom()
    identity = Identity(user_id="u1", agent_id="omni_doll")
    prepared_run = PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context={"blocks": [], "state_summary": ""},
            retrieval_result=RetrievalResponse(memories=[memory, memory]),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="rewritten",
            search_keywords=[],
            worth_saving=True,
            raw_query="hi",
            identity=identity,
            target_topic="topic_1",
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
    )

    await service.finalize_agent_run(prepared_run, AgentRunResult(final_text="done"))

    kernel.librarian_core.lifecycle_engine.record_hit.assert_called_once_with(
        memory.id,
        source="retrieval.finalize",
    )


@pytest.mark.asyncio
async def test_finalize_agent_run_returns_memory_generation_tasks():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.submit_interaction = AsyncMock(return_value=None)
    memory_tasks = [SimpleNamespace(task_id="memtask_1")]
    kernel.librarian_core.run_active_generation = AsyncMock(return_value=memory_tasks)
    service = PatchouliService(runtime=kernel, eye=MagicMock(), global_bus=GlobalSystemBus())

    identity = Identity(user_id="u1", agent_id="omni_doll")
    materialize_task = PendingAtomMaterializeTask(
        pending_alias="pending_fact",
        intent_id="intent_1",
        source_verb="WRITE",
        identity=identity,
        focus=WriteFocus(content="remember this"),
    )
    prepared_run = PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context={"blocks": [], "state_summary": ""},
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="rewritten",
            search_keywords=[],
            worth_saving=True,
            raw_query="hi",
            identity=identity,
            target_topic="topic_1",
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
    )

    result = await service.finalize_agent_run(
        prepared_run,
        AgentRunResult(final_text="done", materialize_tasks=[materialize_task]),
    )

    assert result == memory_tasks
    kernel.librarian_core.run_active_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_agent_run_hit_failure_does_not_fail_finalize():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.submit_interaction = AsyncMock(return_value=None)
    kernel.librarian_core.lifecycle_engine = MagicMock()
    kernel.librarian_core.lifecycle_engine.record_hit.side_effect = RuntimeError("boom")
    service = PatchouliService(runtime=kernel, eye=MagicMock(), global_bus=GlobalSystemBus())

    memory = _build_memory_atom()
    identity = Identity(user_id="u1", agent_id="omni_doll")
    prepared_run = PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context={"blocks": [], "state_summary": ""},
            retrieval_result=RetrievalResponse(memories=[memory]),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="rewritten",
            search_keywords=[],
            worth_saving=True,
            raw_query="hi",
            identity=identity,
            target_topic="topic_1",
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
    )

    await service.finalize_agent_run(prepared_run, AgentRunResult(final_text="done"))

    kernel.librarian_core.submit_interaction.assert_awaited_once()
    kernel.librarian_core.lifecycle_engine.record_hit.assert_called_once()


@pytest.mark.asyncio
async def test_record_memory_citation_calls_lifecycle():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.lifecycle_engine = MagicMock()
    service = PatchouliService(runtime=kernel, eye=MagicMock(), global_bus=GlobalSystemBus())
    memory = _build_memory_atom()
    expected = {"success": True}
    kernel.librarian_core.lifecycle_engine.record_citation.return_value = expected

    result = await service.record_memory_citation(str(memory.id), source="mtp.read")

    assert result is expected
    kernel.librarian_core.lifecycle_engine.record_citation.assert_called_once_with(
        memory.id,
        source="mtp.read",
    )
