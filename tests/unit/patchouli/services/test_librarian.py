from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import Identity, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason, LogicalBlock, TopicMaterializeTask
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.models import TopicData
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.lifecycle import LifecycleFamiliar
from hivememory.patchouli.services.perception import PerceptionFamiliar


class TestPerceptionFamiliar:
    @pytest.mark.asyncio
    async def test_submit_interaction_delegates_to_layer_and_submits_settlement(self):
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
            identity=Identity(user_id="u1"),
        )
        settlement = TopicMaterializeTask(topic_id="t1", blocks=[
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ])
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", settlement))
        layer.settle_topic = AsyncMock(return_value=None)
        layer.prepare_topic = AsyncMock(return_value="t1")
        store = Mock()
        store.topic_exists.return_value = True
        store.needs_eviction.return_value = False
        bus = Mock()
        bus.request = AsyncMock(return_value=None)
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
        )

        result = await familiar.submit_interaction(payload, "t1")

        assert result == "t1"
        layer.route_and_ingest.assert_awaited_once_with("t1", payload)
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )

    @pytest.mark.asyncio
    async def test_manual_settle_returns_none_for_empty_topic(self):
        store = Mock()
        store.get_last_active_topic.return_value = "t1"
        store.get_topic_data.return_value = TopicData(
            topic_id="t1",
            user_id="u1",
            topic_title="empty",
            last_update=1.0,
            last_accessed_at=1.0,
        )
        layer = Mock()
        bus = Mock()
        bus.request = AsyncMock()
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
        )

        result = await familiar.manual_settle_topic()

        assert result is None
        bus.request.assert_not_awaited()


class TestLifecycleFamiliar:
    @pytest.mark.asyncio
    async def test_run_gardening_once_delegates_to_lifecycle_engine(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection.return_value = 3
        familiar = LifecycleFamiliar(lifecycle_engine=lifecycle)

        result = await familiar.run_gardening_once()

        assert result["success"] is True
        assert result["archived_count"] == 3
        lifecycle.run_garbage_collection.assert_called_once_with(force=False)

    @pytest.mark.asyncio
    async def test_run_gardening_once_reports_unavailable_engine(self):
        familiar = LifecycleFamiliar(lifecycle_engine=None)

        result = await familiar.run_gardening_once()

        assert result["success"] is False
        assert result["error"] == "lifecycle_engine is not available"


class TestShortTermMemoryLibraryBoundary:
    def test_topic_data_is_read_view_not_semantic_buffer(self):
        store = ShortTermMemoryStore()
        topic = store.create_buffer("u1")
        store.add_block(
            topic.topic_id,
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),
        )

        data = store.get_topic_data(topic.topic_id)

        assert data is not None
        assert isinstance(data, TopicData)
        assert not hasattr(data, "clear")
        assert data.block_count == 1
