"""Patchouli prepare/finalize 通过 GlobalSystemBus 调用 Alice 的单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, OMNI_DOLL_PROFILE, TurnEvent
from hivememory.core.protocol.models import AgentRunContext, ChatResult, EyeGazeResult, RetrievalResponse
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.contracts.routes import GlobalRoutes
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
async def test_prepare_agent_run_uses_global_bus_for_alias_registration():
    kernel = MagicMock()
    eye = MagicMock()
    bus = GlobalSystemBus()
    local_bus = PatchouliBus()
    kernel.local_bus = local_bus
    kernel.check_storage_health.return_value = True

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

    register_aliases = AsyncMock(return_value=None)
    bus.register(GlobalRoutes.ALICE_REGISTER_PRERETRIEVAL_ALIASES, register_aliases)

    service = PatchouliService(runtime=kernel, eye=eye, global_bus=bus, local_bus=local_bus)

    prepared = await service.prepare_agent_run(
        user_message="hi",
        user_id="u1",
    )

    register_aliases.assert_awaited_once_with(retrieval_result.memories)
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
        loop_result=ChatResult(
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
    assert payload.write_focus is None
    assert payload.update_focus is None
