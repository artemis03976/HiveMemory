import asyncio
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
)
from hivememory.core.protocol.models import EyeGazeResult, RetrievalResponse
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(
            source_agent_id="agent-1",
            user_id="user-1",
            confidence_score=0.9,
            access_count=3,
            vitality_score=88.0,
        ),
        index=IndexLayer(
            title="Python utility function",
            summary="A helper function for date parsing",
            tags=["python", "utils"],
            memory_type=MemoryType.CODE_SNIPPET,
            alias="code_parse_date",
        ),
        payload=PayloadLayer(content="def parse_date(s): return s"),
    )


def _register_prepare_routes(
    bus: PatchouliBus,
    *,
    eye: MagicMock,
    retrieval_result,
) -> None:
    bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, AsyncMock(return_value=OMNI_DOLL_PROFILE))
    bus.register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    bus.register(PatchouliLocalRoutes.GATEWAY_GAZE, eye.gaze)
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, AsyncMock(return_value="topic-1"))
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, AsyncMock(return_value=retrieval_result))
    bus.register(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, AsyncMock(return_value=True))


def test_chat_stream_memory_refs_uses_flatten_schema():
    eye = MagicMock()
    eye.gaze = AsyncMock(
        return_value=EyeGazeResult(
            intent=GatewayIntent.RAG,
            rewritten_query="hello",
            search_keywords=[],
            worth_saving=True,
            raw_query="hello",
            identity=Identity(user_id="user-1"),
            target_topic="NEW_TOPIC",
        )
    )
    local_bus = PatchouliBus()
    _register_prepare_routes(
        local_bus,
        eye=eye,
        retrieval_result=RetrievalResponse(
            rendered_context="",
            memories=[_build_memory_atom()],
        ),
    )
    service = PatchouliService(bus=local_bus, eye=eye)

    prepared = asyncio.run(
        service.prepare_agent_run(
            user_message="hello",
            user_id="user-1",
        )
    )
    memory = prepared.stream_prelude.memory_refs[0]

    assert "title" in memory
    assert "summary" in memory
    assert "memory_type" in memory
    assert "tags" in memory
    assert "content" in memory
    assert "confidence_score" in memory
    assert "index" not in memory
    assert "payload" not in memory
    assert "meta" not in memory


def test_chat_stream_memory_refs_emits_empty_list_when_no_retrieval_hit():
    eye = MagicMock()
    eye.gaze = AsyncMock(
        return_value=EyeGazeResult(
            intent=GatewayIntent.CHAT,
            rewritten_query="hello",
            search_keywords=[],
            worth_saving=True,
            raw_query="hello",
            identity=Identity(user_id="user-1"),
            target_topic="NEW_TOPIC",
        )
    )
    local_bus = PatchouliBus()
    _register_prepare_routes(
        local_bus,
        eye=eye,
        retrieval_result=RetrievalResponse(rendered_context="", memories=[]),
    )
    service = PatchouliService(bus=local_bus, eye=eye)

    prepared = asyncio.run(
        service.prepare_agent_run(
            user_message="hello",
            user_id="user-1",
        )
    )

    assert prepared.stream_prelude.memory_refs == []
