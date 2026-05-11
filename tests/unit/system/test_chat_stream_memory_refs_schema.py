import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.protocol.models import EyeGazeResult, KernelHotResult
from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.worker_agent import StreamChunk, GenerationResult


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
    system = MagicMock(spec=PatchouliSystem)
    system.config = MagicMock()
    system.config.koakuma.max_recursion_depth = 3

    system.eye = MagicMock()
    system.eye.gaze = AsyncMock(
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

    memory_atom = _build_memory_atom()
    system.kernel = MagicMock()
    system.kernel.get_topic_snapshots = AsyncMock(return_value=[])
    system.kernel.prepare_topic = AsyncMock(
        return_value=(
            "topic-1",
            {"topics": [], "max_resident_topics": 5, "current_count": 1},
            {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新话题"},
        )
    )
    system.kernel.handle_hot = AsyncMock(
        return_value=KernelHotResult(
            intent="Chat",
            rewritten="hello",
            keywords=[],
            worth_saving=True,
            rendered_memory_context=None,
            retrieved_memories=[memory_atom],
        )
    )
    system.kernel.submit_interaction = AsyncMock(return_value=None)
    system.kernel.koakuma = MagicMock()
    system.kernel.koakuma.set_current_identity = MagicMock()
    system.kernel.koakuma.reset_interaction_state = MagicMock()
    system.kernel.koakuma.get_interaction_traces = MagicMock(return_value=[])
    system.kernel.koakuma.get_write_focus = MagicMock(return_value=None)
    system.kernel.koakuma.get_update_focus = MagicMock(return_value=None)

    async def fake_stream(_messages):
        yield StreamChunk(
            is_final=True,
            result=GenerationResult(
                text="ok",
                finish_reason="stop",
                was_mtp_interrupted=False,
                prefix_text="ok",
                mtp_fragment="",
            ),
        )

    system._worker_agent = MagicMock()
    system._worker_agent.generate_stream = fake_stream
    system._assemble_messages_from_context = types.MethodType(
        PatchouliSystem._assemble_messages_from_context, system
    )
    system.chat_stream = types.MethodType(PatchouliSystem.chat_stream, system)

    async def _collect():
        events = []
        async for event in system.chat_stream(
            user_message="hello",
            user_id="user-1",
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    memory_event = next(e for e in events if e["event"] == "memory_refs")
    memory = memory_event["data"]["memories"][0]

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
    system = MagicMock(spec=PatchouliSystem)
    system.config = MagicMock()
    system.config.koakuma.max_recursion_depth = 3

    system.eye = MagicMock()
    system.eye.gaze = AsyncMock(
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

    system.kernel = MagicMock()
    system.kernel.get_topic_snapshots = AsyncMock(return_value=[])
    system.kernel.prepare_topic = AsyncMock(
        return_value=(
            "topic-1",
            {"topics": [], "max_resident_topics": 5, "current_count": 1},
            {"state_summary": "", "blocks": [], "total_tokens": 0, "title": "新话题"},
        )
    )
    system.kernel.handle_hot = AsyncMock(
        return_value=KernelHotResult(
            intent="Chat",
            rewritten="hello",
            keywords=[],
            worth_saving=True,
            rendered_memory_context=None,
            retrieved_memories=[],
        )
    )
    system.kernel.submit_interaction = AsyncMock(return_value=None)
    system.kernel.koakuma = MagicMock()
    system.kernel.koakuma.set_current_identity = MagicMock()
    system.kernel.koakuma.reset_interaction_state = MagicMock()
    system.kernel.koakuma.get_interaction_traces = MagicMock(return_value=[])
    system.kernel.koakuma.get_write_focus = MagicMock(return_value=None)
    system.kernel.koakuma.get_update_focus = MagicMock(return_value=None)

    async def fake_stream(_messages):
        yield StreamChunk(
            is_final=True,
            result=GenerationResult(
                text="ok",
                finish_reason="stop",
                was_mtp_interrupted=False,
                prefix_text="ok",
                mtp_fragment="",
            ),
        )

    system._worker_agent = MagicMock()
    system._worker_agent.generate_stream = fake_stream
    system._assemble_messages_from_context = types.MethodType(
        PatchouliSystem._assemble_messages_from_context, system
    )
    system.chat_stream = types.MethodType(PatchouliSystem.chat_stream, system)

    async def _collect():
        events = []
        async for event in system.chat_stream(
            user_message="hello",
            user_id="user-1",
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    memory_event = next(e for e in events if e["event"] == "memory_refs")
    assert memory_event["data"]["memories"] == []
