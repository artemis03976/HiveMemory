"""Patchouli prepare/finalize 通过 GlobalSystemBus 调用 Alice 的单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.core.protocol.models import ChatResult, EyeGazeResult, RetrievalResponse
from hivememory.engines.gateway.models import GatewayIntent
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
        AsyncMock(return_value=MagicMock()),
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
    service._assemble_messages_from_context = MagicMock(return_value=[{"role": "user", "content": "hi"}])

    prepared = await service.prepare_agent_run(
        user_message="hi",
        user_id="u1",
    )

    register_aliases.assert_awaited_once_with(retrieval_result.memories)
    assert prepared.identity.user_id == "u1"
    assert prepared.topic_id == "topic_1"
    assert prepared.stream_prelude.memory_refs[0]["alias"] == "mem_alias"


@pytest.mark.asyncio
async def test_finalize_agent_run_reads_interaction_state_via_global_bus():
    kernel = MagicMock()
    kernel.librarian_core = MagicMock()
    kernel.librarian_core.submit_interaction = AsyncMock(return_value=None)
    service = PatchouliService(
        runtime=kernel,
        eye=MagicMock(),
        global_bus=GlobalSystemBus(),
    )

    interaction_state = {
        "mtp_traces": [],
        "write_focus": None,
        "update_focus": None,
    }
    service._global_bus.register(
        GlobalRoutes.ALICE_GET_INTERACTION_STATE,
        AsyncMock(return_value=interaction_state),
    )

    prepared_run = MagicMock()
    prepared_run.finalize_context = MagicMock(
        user_message="hi",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        gaze_result=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="rewritten",
            search_keywords=[],
            worth_saving=True,
            raw_query="hi",
            identity=Identity(user_id="u1", agent_id="omni_doll"),
            target_topic="topic_1",
        ),
    )

    await service.finalize_agent_run(
        prepared_run=prepared_run,
        loop_result=ChatResult(final_text="done"),
    )

    kernel.librarian_core.submit_interaction.assert_awaited_once()
