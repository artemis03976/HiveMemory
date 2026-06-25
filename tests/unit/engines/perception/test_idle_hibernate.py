import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import Identity, TurnEvent
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.models import FlushReason
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.system.config import SemanticFlowPerceptionConfig


def _make_identity(user="u1", agent="a1"):
    return Identity(user_id=user, agent_id=agent)


def _make_payload(user_msg="hello", assistant_msg="world", identity=None):
    identity = identity or _make_identity()
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        identity=identity,
    )


def _make_familiar(*, idle_timeout_seconds=1, max_resident_topics=5):
    store = ShortTermMemoryStore(max_resident_topics=max_resident_topics)
    relay = Mock()
    relay.should_relay.return_value = None
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
        relay_controller=relay,
        short_term_store=store,
    )
    bus = Mock()
    bus.request = AsyncMock(return_value=None)
    library = MemoryLibrary(short_term=store, mid_term=Mock(), long_term=Mock())
    familiar = PerceptionFamiliar(
        perception_layer=layer,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout_seconds),
        memory_library=library,
    )
    return familiar, layer, store, bus


class TestIdleHibernateSwapOut:
    @pytest.mark.asyncio
    async def test_idle_flush_swaps_out_topic(self):
        familiar, _, store, bus = _make_familiar(idle_timeout_seconds=1)
        await familiar.submit_interaction(_make_payload("question", "answer"), "NEW_TOPIC")
        assert len(store.list_topic_data()) == 1

        time.sleep(1.1)
        flushed = await familiar.scan_idle_buffers_once()

        assert len(flushed) == 1
        assert store.list_topic_data() == []
        bus.request.assert_awaited_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            bus.request.await_args.args[1],
        )
        assert bus.request.await_args.args[1].reason == FlushReason.IDLE_TIMEOUT

    @pytest.mark.asyncio
    async def test_idle_flush_skips_empty_settlement_submission(self):
        familiar, layer, store, bus = _make_familiar(idle_timeout_seconds=1)
        topic_id = await layer.create_new_topic(_make_identity())
        assert store.get_topic_data(topic_id) is not None

        time.sleep(1.1)
        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == [topic_id]
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_flush_frees_slot(self):
        familiar, _, store, _ = _make_familiar(
            idle_timeout_seconds=1,
            max_resident_topics=2,
        )
        await familiar.submit_interaction(_make_payload("q1", "a1", _make_identity("u1", "a1")), "NEW_TOPIC")
        await familiar.submit_interaction(_make_payload("q2", "a2", _make_identity("u2", "a2")), "NEW_TOPIC")
        assert len(store.list_topic_data()) == 2

        time.sleep(1.1)
        assert len(await familiar.scan_idle_buffers_once()) == 2

        await familiar.submit_interaction(_make_payload("q3", "a3", _make_identity("u3", "a3")), "NEW_TOPIC")
        assert len(store.list_topic_data()) == 1

    @pytest.mark.asyncio
    async def test_shutdown_flush_archives_and_swaps_out_all_topics(self):
        familiar, _, store, bus = _make_familiar(max_resident_topics=4)
        await familiar.submit_interaction(_make_payload("q1", "a1", _make_identity("u1", "a1")), "NEW_TOPIC")
        await familiar.submit_interaction(_make_payload("q2", "a2", _make_identity("u2", "a2")), "NEW_TOPIC")

        result = await familiar.flush_all_for_shutdown()

        assert result.trigger_reason == FlushReason.SHUTDOWN.value
        assert len(result.flushed_topics) == 2
        assert result.archived_blocks == 2
        assert store.list_topic_data() == []
        assert bus.request.await_count == 2
        for call in bus.request.await_args_list:
            assert call.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
            assert call.args[1].reason == FlushReason.SHUTDOWN
