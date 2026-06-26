"""
PerceptionFamiliar 单元测试

测试覆盖:
- submit_interaction: 交互摄入与 settlement 提交
- manual_settle_topic: 手动结算
- evict_topic: 话题驱逐
- discard_if_empty: 空话题清理
- scan_idle_buffers_once: 空闲话题扫描与结算
"""

import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from hivememory.core.models import Identity, TurnEvent, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason, LogicalBlock, TopicMaterializeTask
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.models import TopicData
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


def _make_real_familiar(*, idle_timeout_seconds=1, max_resident_topics=5):
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


class TestPerceptionFamiliar:
    """PerceptionFamiliar 完整测试套件"""

    def _make_familiar(self, layer=None, store=None, bus=None, idle_timeout=30):
        layer = layer or Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", None))
        layer.prepare_topic = AsyncMock(return_value="t1")
        layer.settle_topic = AsyncMock(return_value=None)
        layer.swap_out_topic = Mock(return_value=True)
        layer.discard_if_empty = Mock(return_value=True)

        store = store or Mock()
        store.topic_exists = Mock(return_value=True)
        store.needs_eviction = Mock(return_value=False)
        store.get_lru_buffer = Mock(return_value=None)
        store.list_topic_data = Mock(return_value=[])
        store.get_last_active_topic = Mock(return_value="t1")
        store.get_topic_data = Mock(return_value=TopicData(
            topic_id="t1",
            user_id="u1",
            topic_title="title",
            last_update=1.0,
            last_accessed_at=1.0,
        ))

        bus = bus or Mock()
        bus.request = AsyncMock(return_value=None)

        memory_lib = Mock()
        memory_lib.short_term = store

        return PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=idle_timeout),
            memory_library=memory_lib,
        )

    @pytest.mark.asyncio
    async def test_submit_interaction_delegates_to_layer_and_submits_settlement(self):
        """验证 submit_interaction 正确调用 layer.route_and_ingest 并提交 settlement"""
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
    async def test_submit_interaction_no_settlement_when_route_returns_none(self):
        """验证当 route_and_ingest 返回 None settlement 时，不调用 bus.request"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
            identity=Identity(user_id="u1"),
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", None))  # 无 settlement
        store = Mock()
        store.topic_exists.return_value = True
        store.needs_eviction.return_value = False

        bus = Mock()
        bus.request = AsyncMock()

        familiar = self._make_familiar(layer=layer, store=store, bus=bus)

        result = await familiar.submit_interaction(payload, "t1")

        assert result == "t1"
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_interaction_evicts_lru_before_new_topic_when_pool_full(self):
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
            identity=Identity(user_id="u1"),
        )
        lru = TopicData(
            topic_id="old_topic",
            user_id="u1",
            topic_title="old",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="old", assistant_final_text="answer")),),
            last_update=1.0,
            last_accessed_at=1.0,
        )
        settlement = TopicMaterializeTask(
            topic_id="old_topic",
            blocks=[LogicalBlock(turn=TurnRecord(user_query="old", assistant_final_text="answer"))],
            reason=FlushReason.LRU_EVICTION,
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("new_topic", None))
        layer.settle_topic = AsyncMock(return_value=settlement)
        store = Mock()
        store.topic_exists.return_value = False
        store.needs_eviction.return_value = True
        store.get_lru_topic.return_value = "old_topic"
        bus = Mock()
        bus.request = AsyncMock(return_value=None)
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
        )

        result = await familiar.submit_interaction(payload, "NEW_TOPIC")

        assert result == "new_topic"
        layer.settle_topic.assert_awaited_once_with("old_topic", FlushReason.LRU_EVICTION)
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )
        layer.route_and_ingest.assert_awaited_once_with("NEW_TOPIC", payload)

    @pytest.mark.asyncio
    async def test_manual_settle_returns_none_for_empty_topic(self):
        """验证手动结算空话题时返回 None"""
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

    @pytest.mark.asyncio
    async def test_manual_settle_returns_task_for_non_empty_topic(self):
        """验证手动结算非空话题时返回任务"""
        from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
        store = Mock()
        store.get_last_active_topic.return_value = "t1"
        store.get_topic_data.return_value = TopicData(
            topic_id="t1",
            user_id="u1",
            topic_title="title",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            last_update=1.0,
            last_accessed_at=1.0,
        )
        layer = Mock()
        layer.settle_topic = AsyncMock(return_value=TopicMaterializeTask(
            topic_id="t1",
            topic_title="title",
            topic_summary="",
            blocks=[],
            state_summary="",
        ))
        bus = Mock()
        expected_task = MemoryGenerationTask(
            task_id="task-1",
            topic_id="t1",
            label="t1",
            source=Mock(value="ARCHIVE"),
        )
        bus.request = AsyncMock(return_value=expected_task)
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
        )

        result = await familiar.manual_settle_topic()

        assert result is expected_task
        bus.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evict_topic_calls_layer_swap_out(self):
        """验证 evict_topic 正确调用 layer.swap_out_topic"""
        layer = Mock()
        layer.swap_out_topic = Mock(return_value=True)
        familiar = self._make_familiar(layer=layer)

        result = await familiar.evict_topic("topic_to_evict")

        assert result.success is True
        layer.swap_out_topic.assert_called_once_with("topic_to_evict")

    def test_discard_if_empty_delegates_to_layer(self):
        """验证 discard_if_empty 正确委托给 layer"""
        layer = Mock()
        layer.discard_if_empty = Mock(return_value=True)
        familiar = self._make_familiar(layer=layer)

        result = familiar.discard_if_empty("empty_topic")

        assert result is True
        layer.discard_if_empty.assert_called_once_with("empty_topic")

    @pytest.mark.asyncio
    async def test_scan_idle_buffers_once_skips_non_idle_topics(self):
        """验证空闲扫描跳过非空闲话题"""
        from datetime import datetime
        recent_topic_data = TopicData(
            topic_id="recent_topic",
            user_id="u1",
            topic_title="recent",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            last_update=datetime.now().timestamp(),  # 刚刚更新
            last_accessed_at=datetime.now().timestamp(),
        )
        layer = Mock()
        layer.settle_topic = AsyncMock()
        store = Mock()
        store.list_topic_data = Mock(return_value=[recent_topic_data])

        bus = Mock()
        familiar = self._make_familiar(layer=layer, store=store, bus=bus)
        familiar._idle_timeout_seconds = 3600  # 1小时

        flushed = await familiar.scan_idle_buffers_once()

        assert len(flushed) == 0
        layer.settle_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_flush_swaps_out_topic(self):
        familiar, _, store, bus = _make_real_familiar(idle_timeout_seconds=1)
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
        familiar, layer, store, bus = _make_real_familiar(idle_timeout_seconds=1)
        topic_id = await layer.create_new_topic(_make_identity())
        assert store.get_topic_data(topic_id) is not None

        time.sleep(1.1)
        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == [topic_id]
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_flush_frees_slot(self):
        familiar, _, store, _ = _make_real_familiar(
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
        familiar, _, store, bus = _make_real_familiar(max_resident_topics=4)
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
