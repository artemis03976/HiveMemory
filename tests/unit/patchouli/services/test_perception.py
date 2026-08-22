"""
PerceptionFamiliar 单元测试

测试覆盖:
- submit_interaction: 交互摄入与 settlement 提交（mock 边界）
- manual_settle_topic: 手动结算
- evict_topic: 话题驱逐
- discard_if_empty: 空话题清理
- scan_idle_buffers_once: 空闲话题扫描与结算

真实链路（PerceptionFamiliar + Layer + Store + TriggerManager 协作）测试
位于 tests/integration/patchouli/test_perception_flush_chain.py。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import LogicalBlock, TopicData, TurnRecord, WorkspaceTopicKey
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason, TopicMaterializeTask
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.errors import TopicSettleAdmissionError
from hivememory.patchouli.services.perception import PerceptionFamiliar
from tests.helpers.memory import make_memory_creation_context
from tests.helpers.workspace import make_access_context


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
            workspace_identity=make_access_context(user_id="u1").workspace_identity,
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
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

    @pytest.mark.asyncio
    async def test_submit_interaction_delegates_to_layer_and_submits_settlement(self):
        """验证 submit_interaction 正确调用 layer.route_and_ingest 并提交 settlement"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        settlement = TopicMaterializeTask(
            topic_id="t1",
            identity_scope=make_memory_creation_context(user_id="u1"),
            blocks=[LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))],
        )
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
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

        result = await familiar.submit_interaction(
            payload,
            identity_scope=make_access_context(user_id="u1"),
            target_topic_id="t1",
        )

        assert result == "t1"
        layer.route_and_ingest.assert_awaited_once_with(
            "t1",
            payload,
            identity_scope=make_access_context(user_id="u1"),
        )
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
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", None))  # 无 settlement
        store = Mock()
        store.topic_exists.return_value = True
        store.needs_eviction.return_value = False

        bus = Mock()
        bus.request = AsyncMock()

        familiar = self._make_familiar(layer=layer, store=store, bus=bus)

        result = await familiar.submit_interaction(
            payload,
            identity_scope=make_access_context(user_id="u1"),
            target_topic_id="t1",
        )

        assert result == "t1"
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_interaction_evicts_lru_before_new_topic_when_pool_full(self):
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        settlement = TopicMaterializeTask(
            topic_id="old_topic",
            identity_scope=make_memory_creation_context(user_id="u1"),
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
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

        result = await familiar.submit_interaction(
            payload,
            identity_scope=make_access_context(user_id="u1"),
            target_topic_id="NEW_TOPIC",
        )

        assert result == "new_topic"
        layer.settle_topic.assert_awaited_once_with(
            WorkspaceTopicKey.from_access_context(make_access_context(user_id="u1"), "old_topic"),
            FlushReason.LRU_EVICTION,
        )
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )
        layer.route_and_ingest.assert_awaited_once_with(
            "NEW_TOPIC",
            payload,
            identity_scope=make_access_context(user_id="u1"),
        )

    @pytest.mark.asyncio
    async def test_manual_settle_evicts_empty_topic_without_generation(self):
        """真正空 Topic 无可提交材料：不触发生成，但仍结束生命周期并返回成功。"""
        store = Mock()
        store.get_last_active_topic.return_value = "t1"
        store.get_topic_data.return_value = TopicData(
            topic_id="t1",
            workspace_identity=make_access_context(user_id="u1").workspace_identity,
            topic_title="empty",
            last_update=1.0,
            last_accessed_at=1.0,
        )
        layer = Mock()
        layer.prepare_settlement = AsyncMock(return_value=None)
        layer.swap_out_topic = Mock(return_value=True)
        bus = Mock()
        bus.request = AsyncMock()
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

        result = await familiar.manual_settle_topic(make_access_context(user_id="u1"))

        assert result.topic_id == "t1"
        assert result.generation_submitted is False
        assert result.generation_task_id is None
        bus.request.assert_not_awaited()
        layer.swap_out_topic.assert_called_once_with(
            WorkspaceTopicKey.from_access_context(make_access_context(user_id="u1"), "t1")
        )

    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_evicts(self):
        """manual settle：先接纳 generation task，成功后才驱逐 Topic。"""
        from hivememory.patchouli.control.memory_generation.models import (
            MemoryGenerationSource,
            MemoryGenerationTask,
        )
        store = Mock()
        store.get_last_active_topic.return_value = "t1"
        store.get_topic_data.return_value = TopicData(
            topic_id="t1",
            workspace_identity=make_access_context(user_id="u1").workspace_identity,
            topic_title="title",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            last_update=1.0,
            last_accessed_at=1.0,
        )
        layer = Mock()
        layer.prepare_settlement = AsyncMock(return_value=TopicMaterializeTask(
            topic_id="t1",
            identity_scope=make_memory_creation_context(user_id="u1"),
            topic_title="title",
            topic_summary="",
            blocks=[],
            state_summary="",
        ))
        layer.swap_out_topic = Mock(return_value=True)
        bus = Mock()
        expected_task = MemoryGenerationTask(
            task_id="task-1",
            topic_id="t1",
            label="t1",
            source=MemoryGenerationSource.SETTLE,
        )
        bus.request = AsyncMock(return_value=expected_task)
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

        access_context = make_access_context(user_id="u1")
        result = await familiar.manual_settle_topic(access_context)

        assert result.topic_id == "t1"
        assert result.generation_task_id == "task-1"
        assert result.generation_submitted is True
        layer.prepare_settlement.assert_awaited_once_with(
            WorkspaceTopicKey.from_access_context(access_context, "t1")
        )
        assert (
            bus.request.await_args.args[0]
            == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        )
        layer.swap_out_topic.assert_called_once_with(
            WorkspaceTopicKey.from_access_context(access_context, "t1")
        )

    @pytest.mark.asyncio
    async def test_manual_settle_admission_failure_keeps_topic_intact(self):
        """admission 失败：抛出受控错误，且 Topic 不被驱逐、材料不被清空。"""
        store = Mock()
        store.get_last_active_topic.return_value = "t1"
        store.get_topic_data.return_value = TopicData(
            topic_id="t1",
            workspace_identity=make_access_context(user_id="u1").workspace_identity,
            topic_title="title",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            last_update=1.0,
            last_accessed_at=1.0,
        )
        layer = Mock()
        layer.prepare_settlement = AsyncMock(return_value=TopicMaterializeTask(
            topic_id="t1",
            identity_scope=make_memory_creation_context(user_id="u1"),
            topic_title="title",
            blocks=[LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))],
        ))
        layer.swap_out_topic = Mock(return_value=True)
        bus = Mock()
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            bus=bus,
            config=SimpleNamespace(idle_timeout_seconds=30),
            memory_library=SimpleNamespace(short_term=store),
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

        with pytest.raises(TopicSettleAdmissionError, match="可重试"):
            await familiar.manual_settle_topic(make_access_context(user_id="u1"))

        layer.swap_out_topic.assert_not_called()
        store.get_topic_data.assert_called()
        store.clear_blocks.assert_not_called()

    @pytest.mark.asyncio
    async def test_evict_topic_calls_layer_swap_out(self):
        """验证 evict_topic 正确调用 layer.swap_out_topic"""
        layer = Mock()
        layer.swap_out_topic = Mock(return_value=True)
        familiar = self._make_familiar(layer=layer)

        access_context = make_access_context(user_id="u1")
        result = await familiar.evict_topic(access_context, "topic_to_evict")

        assert result.topic_id == "topic_to_evict"
        assert result.removed is True
        layer.swap_out_topic.assert_called_once_with(
            WorkspaceTopicKey.from_access_context(access_context, "topic_to_evict")
        )

    @pytest.mark.asyncio
    async def test_scan_idle_buffers_once_skips_non_idle_topics(self):
        """验证空闲扫描跳过非空闲话题"""
        from datetime import datetime
        recent_topic_data = TopicData(
            topic_id="recent_topic",
            workspace_identity=make_access_context(user_id="u1").workspace_identity,
            topic_title="recent",
            blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
            last_update=datetime.now().timestamp(),  # 刚刚更新
            last_accessed_at=datetime.now().timestamp(),
        )
        layer = Mock()
        layer.settle_topic = AsyncMock()
        store = Mock()
        store.list_all_topic_data_for_maintenance = Mock(return_value=[recent_topic_data])

        bus = Mock()
        familiar = self._make_familiar(layer=layer, store=store, bus=bus, idle_timeout=3600)

        flushed = await familiar.scan_idle_buffers_once()

        assert len(flushed) == 0
        layer.settle_topic.assert_not_called()
