"""Patchouli prepare/finalize workflow tests against local route primitives."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
    TurnEvent,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    EyeGazeResult,
    RetrievalResponse,
)
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService


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


def _gaze_result(identity: Identity) -> EyeGazeResult:
    return EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="rewritten",
        search_keywords=["tag"],
        worth_saving=True,
        raw_query="hi",
        identity=identity,
        target_topic="topic_1",
    )


def _prepared_run(
    *,
    identity: Identity | None = None,
    retrieval_result: RetrievalResponse | None = None,
) -> PreparedAgentRun:
    identity = identity or Identity(user_id="u1", agent_id="omni_doll")
    return PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context=None,
            retrieval_result=retrieval_result or RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=_gaze_result(identity),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_topics=[],
            memory_refs=[],
        ),
    )


@pytest.mark.asyncio
async def test_prepare_agent_run_returns_agent_run_context_with_retrieval_result():
    identity = Identity(user_id="u1", agent_id="omni_doll")
    bus = PatchouliBus()
    eye = MagicMock()
    retrieval_result = RetrievalResponse(
        memories=[_build_memory_atom()],
        rendered_context="<memory>ctx</memory>",
    )

    bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, AsyncMock(return_value=OMNI_DOLL_PROFILE))
    bus.register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    bus.register(PatchouliLocalRoutes.GATEWAY_GAZE, AsyncMock(return_value=_gaze_result(identity)))
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, AsyncMock(return_value="topic_1"))
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, AsyncMock(return_value=retrieval_result))
    bus.register(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, AsyncMock(return_value=True))

    service = PatchouliService(bus=bus, eye=eye)

    prepared = await service.prepare_agent_run(user_message="hi", user_id="u1")

    assert prepared.identity.user_id == "u1"
    assert prepared.topic_id == "topic_1"
    assert isinstance(prepared.agent_run_context, AgentRunContext)
    assert prepared.agent_run_context.retrieval_result is retrieval_result
    assert prepared.agent_run_context.topic_context is None
    assert prepared.stream_prelude.pool_topics == []
    assert prepared.stream_prelude.memory_refs[0]["alias"] == "mem_alias"


@pytest.mark.asyncio
async def test_finalize_agent_run_submits_interaction_payload_to_local_route():
    bus = PatchouliBus()
    submit_interaction = AsyncMock(return_value=None)
    bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, submit_interaction)
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, AsyncMock())
    service = PatchouliService(bus=bus, eye=MagicMock())

    await service.finalize_agent_run(
        prepared_run=_prepared_run(),
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

    submit_interaction.assert_awaited_once()
    payload = submit_interaction.await_args.args[0]
    assert submit_interaction.await_args.kwargs == {"target_topic_id": "topic_1"}
    assert payload.mtp_traces
    assert payload.mtp_traces[0].action == "SEARCH"
    assert payload.materialize_tasks == []


@pytest.mark.asyncio
async def test_finalize_agent_run_records_retrieval_hits_once_per_memory():
    bus = PatchouliBus()
    record_hit = AsyncMock()
    memory = _build_memory_atom()
    bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    service = PatchouliService(bus=bus, eye=MagicMock())

    await service.finalize_agent_run(
        _prepared_run(retrieval_result=RetrievalResponse(memories=[memory, memory])),
        AgentRunResult(final_text="done"),
    )

    record_hit.assert_awaited_once_with(memory.id, source="retrieval.finalize")


@pytest.mark.asyncio
async def test_finalize_agent_run_returns_active_memory_generation_tasks():
    bus = PatchouliBus()
    memory_tasks = [SimpleNamespace(task_id="memtask_1")]
    submit_active = AsyncMock(return_value=memory_tasks)
    bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, submit_active)
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, AsyncMock())
    service = PatchouliService(bus=bus, eye=MagicMock())

    identity = Identity(user_id="u1", agent_id="omni_doll")
    materialize_task = PendingAtomMaterializeTask(
        pending_alias="pending_fact",
        intent_id="intent_1",
        source_verb="WRITE",
        identity=identity,
        focus=WriteFocus(content="remember this"),
    )

    result = await service.finalize_agent_run(
        _prepared_run(identity=identity),
        AgentRunResult(final_text="done", materialize_tasks=[materialize_task]),
    )

    assert result == memory_tasks
    submit_active.assert_awaited_once_with([materialize_task], topic_id="topic_1")


@pytest.mark.asyncio
async def test_finalize_agent_run_hit_failure_does_not_fail_finalize():
    bus = PatchouliBus()
    record_hit = AsyncMock(side_effect=RuntimeError("boom"))
    memory = _build_memory_atom()
    bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    service = PatchouliService(bus=bus, eye=MagicMock())

    result = await service.finalize_agent_run(
        _prepared_run(retrieval_result=RetrievalResponse(memories=[memory])),
        AgentRunResult(final_text="done"),
    )

    assert result == []
    record_hit.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_memory_citation_calls_local_lifecycle_route():
    bus = PatchouliBus()
    memory = _build_memory_atom()
    expected = {"success": True}
    record_citation = AsyncMock(return_value=expected)
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_CITATION, record_citation)
    service = PatchouliService(bus=bus, eye=MagicMock())

    result = await service.record_memory_citation(str(memory.id), source="mtp.read")

    assert result is expected
    record_citation.assert_awaited_once_with(memory.id, source="mtp.read")
