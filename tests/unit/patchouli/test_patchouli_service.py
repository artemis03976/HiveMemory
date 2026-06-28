import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import (
    EyeGazeResult,
    RetrievalResponse,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService


def _make_gaze_result(intent=GatewayIntent.RAG, **kwargs):
    defaults = dict(
        intent=intent,
        rewritten_query="重写查询",
        search_keywords=["kw1"],
        worth_saving=True,
        raw_query="原始查询",
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        target_topic="topic-1",
    )
    defaults.update(kwargs)
    return EyeGazeResult(**defaults)


def _make_retrieval_response(empty=False):
    return RetrievalResponse(
        memories=[],
    )


@pytest.fixture
def service_with_local_bus():
    local_bus = PatchouliBus()
    service = PatchouliService(
        bus=local_bus,
        eye=MagicMock(),
    )
    return service


@pytest.mark.asyncio
async def test_retrieve_for_gaze_rag_intent(service_with_local_bus):
    service = service_with_local_bus
    gaze = _make_gaze_result(intent=GatewayIntent.RAG)
    retrieved = _make_retrieval_response(empty=False)
    retrieve = AsyncMock(return_value=retrieved)
    service._local_bus.register("memory.retrieve", retrieve)

    result = await service.retrieve_for_gaze(gaze)

    assert isinstance(result, RetrievalResponse)
    assert not hasattr(result, "rendered_context")
    retrieve.assert_awaited_once()
    request = retrieve.await_args.args[0]
    assert request.semantic_query == "重写查询"
    assert request.keywords == ["kw1"]
    assert request.user_id == "u1"


@pytest.mark.asyncio
async def test_retrieve_for_gaze_chat_intent_skips_retrieval(service_with_local_bus):
    service = service_with_local_bus
    gaze = _make_gaze_result(intent=GatewayIntent.CHAT)
    retrieve = AsyncMock()
    service._local_bus.register("memory.retrieve", retrieve)

    result = await service.retrieve_for_gaze(gaze)

    assert result.memories == []
    retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_for_gaze_retrieval_disabled(service_with_local_bus):
    service = service_with_local_bus
    gaze = _make_gaze_result(intent=GatewayIntent.RAG)
    retrieve = AsyncMock()
    service._local_bus.register("memory.retrieve", retrieve)

    result = await service.retrieve_for_gaze(gaze, enable_retrieval=False)

    assert result.memories == []
    retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_for_gaze_empty_retrieval(service_with_local_bus):
    service = service_with_local_bus
    gaze = _make_gaze_result(intent=GatewayIntent.RAG)
    retrieve = AsyncMock(return_value=_make_retrieval_response(empty=True))
    service._local_bus.register("memory.retrieve", retrieve)

    result = await service.retrieve_for_gaze(gaze)

    assert result.memories == []
