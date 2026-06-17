import asyncio
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, OMNI_DOLL_PROFILE
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import EyeGazeResult, RetrievalResponse
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


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
            title="Python 工具函数",
            summary="这是一个用于日期解析的工具函数摘要文本",
            tags=["python", "utils"],
            memory_type=MemoryType.CODE_SNIPPET,
            alias="code_parse_date",
        ),
        payload=PayloadLayer(
            content="def parse_date(s): return s",
        ),
    )


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

    memory_atom = _build_memory_atom()
    kernel = MagicMock()
    bus = GlobalSystemBus()
    local_bus = PatchouliBus()
    kernel.local_bus = local_bus
    kernel.check_storage_health = AsyncMock(return_value=True)
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
        AsyncMock(
            return_value=(
                "topic-1",
                {"topics": [], "max_resident_topics": 5, "current_count": 1},
                {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新话题"},
            )
        ),
    )
    local_bus.register(
        "memory.retrieve",
        AsyncMock(
            return_value=RetrievalResponse(
                rendered_context="",
                memories=[memory_atom],
            )
        ),
    )
    service = PatchouliService(runtime=kernel, eye=eye, global_bus=bus, local_bus=local_bus)

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

    kernel = MagicMock()
    local_bus = PatchouliBus()
    kernel.local_bus = local_bus
    kernel.check_storage_health = AsyncMock(return_value=True)
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
        AsyncMock(
            return_value=(
                "topic-1",
                {"topics": [], "max_resident_topics": 5, "current_count": 1},
                {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新话题"},
            )
        ),
    )
    local_bus.register(
        "memory.retrieve",
        AsyncMock(
            return_value=MagicMock(
                is_empty=MagicMock(return_value=True),
                rendered_context=None,
                memories=[],
            )
        ),
    )
    service = PatchouliService(runtime=kernel, eye=eye, local_bus=local_bus)

    prepared = asyncio.run(
        service.prepare_agent_run(
            user_message="hello",
            user_id="user-1",
        )
    )
    assert prepared.stream_prelude.memory_refs == []
