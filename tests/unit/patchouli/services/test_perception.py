"""
PerceptionFamiliar 单元测试

测试覆盖:
- apply_interaction: 交互应用与 settlement 提交（mock 领域服务边界）
- _maybe_evict_lru: LRU 驱逐与统一 settle 时序
- manual_settle_topic: 手动结算（admission 成功/无材料/失败恢复）
- evict_topic / discard_if_empty: 话题驱逐与空话题清理
- scan_idle_buffers_once / flush_all_for_shutdown: 维护与 shutdown 投影

真实链路（PerceptionFamiliar + Layer + TopicBufferService + Store 协作）测试
位于 tests/integration/patchouli/test_perception_flush_chain.py。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import LogicalBlock, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.services.topic_buffer import (
    SettlementStatus,
    SettlementReservation,
)
from hivememory.patchouli.services.perception import PerceptionFamiliar
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1"):
    return make_identity_scope(user_id=user_id)


def _task(topic_id="t1", reason=TriggerReason.IDLE_TIMEOUT) -> TopicMaterializeTask:
    return TopicMaterializeTask(
        topic_id=topic_id,
        identity_scope=_identity_scope(),
        blocks=(LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a")),),
        reason=reason,
    )


def _reservation(topic_id="t1", task=None) -> SettlementReservation:
    return SettlementReservation(topic_id=topic_id, reason=TriggerReason.IDLE_TIMEOUT, task=task)


def _accepted_outcome(topic_id="t1", *, task_id=None):
    from hivememory.patchouli.services.topic_buffer import SettlementOutcome

    if task_id is not None:
        return SettlementOutcome(
            topic_id=topic_id, status=SettlementStatus.ACCEPTED, removed=True,
            generation_task_id=task_id,
        )
    return SettlementOutcome(
        topic_id=topic_id, status=SettlementStatus.NO_MATERIAL, removed=True,
    )


def _generation_task(task_id="task-1", topic_id="t1") -> MemoryGenerationTask:
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.SETTLE,
    )


class TestPerceptionFamiliar:
    """PerceptionFamiliar 完整测试套件"""

    def _make_familiar(self, *, layer=None, topic_buffer=None, bus=None, idle_timeout=30):
        if layer is None:
            layer = Mock()
            layer.route_and_ingest = AsyncMock(return_value=("t1", None))
            layer.prepare_topic = AsyncMock(return_value="t1")

        topic_buffer = topic_buffer or Mock()
        topic_buffer.touch_topic = Mock(return_value=None)
        topic_buffer.get_topic = Mock(return_value=Mock())  # 默认目标存在
        topic_buffer.count_topics = Mock(return_value=0)
        topic_buffer.select_lru_candidate = Mock(return_value=None)
        topic_buffer.begin_settlement = Mock(return_value=None)
        topic_buffer.complete_settlement = Mock(return_value=_accepted_outcome())
        topic_buffer.abort_settlement = Mock(return_value=None)
        topic_buffer.delete_if_idle = Mock(return_value=True)
        topic_buffer.discard_if_empty = Mock(return_value=True)
        topic_buffer.list_idle_candidates = Mock(return_value=[])
        topic_buffer.list_shutdown_candidates = Mock(return_value=[])

        bus = bus or Mock()
        bus.request = AsyncMock(return_value=None)

        config = SimpleNamespace(
            idle_timeout_seconds=idle_timeout,
            engine=SimpleNamespace(max_resident_topics=5),
        )
        familiar = PerceptionFamiliar(
            perception_layer=layer,
            topic_buffer=topic_buffer,
            bus=bus,
            config=config,
            interaction_journal=InMemoryInteractionApplyJournal(),
        )
        return familiar, topic_buffer, bus

    @pytest.mark.asyncio
    async def test_apply_interaction_delegates_to_layer_and_submits_settlement(self):
        """验证 apply_interaction 正确调用 layer.route_and_ingest 并提交 settlement"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        settlement = _task("t1", TriggerReason.TOKEN_OVERFLOW)
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", settlement))
        familiar, topic_buffer, bus = self._make_familiar(layer=layer)
        topic_buffer.touch_topic.return_value = Mock()

        result = await familiar.apply_interaction(
            payload,
            identity_scope=_identity_scope(),
            target_topic_id="t1",
        )

        assert result == "t1"
        layer.route_and_ingest.assert_awaited_once_with(
            "t1",
            payload,
            identity_scope=_identity_scope(),
            asset_id_and_refs=(),
        )
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )
        topic_buffer.touch_topic.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_interaction_no_settlement_when_route_returns_none(self):
        """验证当 route_and_ingest 返回 None settlement 时，不调用 bus.request"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("t1", None))  # 无 settlement
        familiar, _, bus = self._make_familiar(layer=layer)

        result = await familiar.apply_interaction(
            payload,
            identity_scope=_identity_scope(),
            target_topic_id="t1",
        )

        assert result == "t1"
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_interaction_evicts_lru_before_new_topic_when_pool_full(self):
        """池满时先驱逐 LRU 候选：admission 成功后 complete 释放容量。"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        settlement = _task("old_topic", TriggerReason.LRU_EVICTION)
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("new_topic", None))
        familiar, topic_buffer, bus = self._make_familiar(layer=layer)

        topic_buffer.count_topics.side_effect = [5, 5, 4]  # 池满 -> 驱逐后释放
        topic_buffer.select_lru_candidate.return_value = "old_topic"
        topic_buffer.begin_settlement.return_value = _reservation("old_topic", settlement)
        accepted_task = _generation_task("task-1", "old_topic")
        bus.request = AsyncMock(return_value=accepted_task)
        topic_buffer.complete_settlement.return_value = _accepted_outcome(
            "old_topic", task_id="task-1"
        )

        result = await familiar.apply_interaction(
            payload,
            identity_scope=_identity_scope(),
            target_topic_id="NEW_TOPIC",
        )

        assert result == "new_topic"
        topic_buffer.select_lru_candidate.assert_called_once()
        topic_buffer.begin_settlement.assert_called_once()
        assert topic_buffer.begin_settlement.call_args.args[2] is TriggerReason.LRU_EVICTION
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )
        topic_buffer.complete_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_lru_reselects_when_candidate_busy(self):
        """候选在选择后 busy：改选其他 IDLE 候选。"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("new_topic", None))
        familiar, topic_buffer, bus = self._make_familiar(layer=layer)

        topic_buffer.count_topics.side_effect = [5, 5, 5, 4]
        topic_buffer.select_lru_candidate.side_effect = ["busy_topic", "idle_topic"]
        topic_buffer.begin_settlement.side_effect = [
            TopicBusyError("busy"),
            _reservation("idle_topic", None),
        ]
        topic_buffer.complete_settlement.return_value = _accepted_outcome("idle_topic")

        result = await familiar.apply_interaction(
            payload,
            identity_scope=_identity_scope(),
            target_topic_id="NEW_TOPIC",
        )

        assert result == "new_topic"
        # busy 候选被改选：begin 共两次（busy 一次 + 成功一次）。
        assert topic_buffer.begin_settlement.call_count == 2

    @pytest.mark.asyncio
    async def test_lru_admission_failure_restores_topic_and_raises(self):
        """LRU admission 失败：abort 恢复 Topic（不释放容量），异常向上传播。"""
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )
        layer = Mock()
        layer.route_and_ingest = AsyncMock(return_value=("new_topic", None))
        familiar, topic_buffer, bus = self._make_familiar(layer=layer)

        topic_buffer.count_topics.side_effect = [5, 5]
        topic_buffer.select_lru_candidate.return_value = "old_topic"
        topic_buffer.begin_settlement.return_value = _reservation(
            "old_topic", _task("old_topic", TriggerReason.LRU_EVICTION)
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(RuntimeError, match="admission boom"):
            await familiar.apply_interaction(
                payload,
                identity_scope=_identity_scope(),
                target_topic_id="NEW_TOPIC",
            )

        topic_buffer.abort_settlement.assert_called_once()
        assert topic_buffer.abort_settlement.call_args.kwargs["reason"] is TriggerReason.LRU_EVICTION
        topic_buffer.complete_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_settle_evicts_empty_topic_without_generation(self):
        """真正空 Topic 无可提交材料：不触发生成，但仍结束生命周期并返回成功。"""
        familiar, topic_buffer, bus = self._make_familiar()
        topic_buffer.get_topic.return_value = Mock()  # 目标存在
        topic_buffer.begin_settlement.return_value = _reservation("t1", task=None)
        topic_buffer.complete_settlement.return_value = _accepted_outcome("t1")

        result = await familiar.manual_settle_topic(_identity_scope(), "t1")

        assert result.topic_id == "t1"
        assert result.generation_submitted is False
        assert result.generation_task_id is None
        bus.request.assert_not_awaited()
        topic_buffer.complete_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_completes(self):
        """manual settle：先接纳 generation task，成功后才结束 Topic 生命周期。"""
        familiar, topic_buffer, bus = self._make_familiar()
        topic_buffer.get_topic.return_value = Mock()
        settlement = _task("t1", TriggerReason.MANUAL_SETTLE)
        topic_buffer.begin_settlement.return_value = _reservation("t1", settlement)
        expected_task = _generation_task("task-1", "t1")
        bus.request = AsyncMock(return_value=expected_task)
        topic_buffer.complete_settlement.return_value = _accepted_outcome(
            "t1", task_id="task-1"
        )

        result = await familiar.manual_settle_topic(_identity_scope(), "t1")

        assert result.topic_id == "t1"
        assert result.generation_task_id == "task-1"
        assert result.generation_submitted is True
        assert topic_buffer.begin_settlement.call_args.args[2] is TriggerReason.MANUAL_SETTLE
        assert (
            bus.request.await_args.args[0]
            == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        )
        topic_buffer.complete_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_settle_admission_failure_keeps_topic_intact(self):
        """admission 失败：抛出受控错误，且 Topic 恢复 IDLE、材料不被清空。"""
        familiar, topic_buffer, bus = self._make_familiar()
        topic_buffer.get_topic.return_value = Mock()
        topic_buffer.begin_settlement.return_value = _reservation(
            "t1", _task("t1", TriggerReason.MANUAL_SETTLE)
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(TopicSettleAdmissionError, match="可重试"):
            await familiar.manual_settle_topic(_identity_scope(), "t1")

        topic_buffer.abort_settlement.assert_called_once()
        assert topic_buffer.abort_settlement.call_args.kwargs["reason"] is TriggerReason.MANUAL_SETTLE
        topic_buffer.complete_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_settle_busy_topic_raises_busy_error(self):
        """目标正忙是瞬态冲突：保持 TopicBusyError 语义，不伪装成 admission 失败。"""
        familiar, topic_buffer, _ = self._make_familiar()
        topic_buffer.get_topic.return_value = Mock()
        topic_buffer.begin_settlement.side_effect = TopicBusyError("busy")

        with pytest.raises(TopicBusyError):
            await familiar.manual_settle_topic(_identity_scope(), "t1")

        topic_buffer.abort_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_evict_topic_calls_service_delete_if_idle(self):
        """验证 evict_topic 正确调用领域服务 delete_if_idle"""
        familiar, topic_buffer, _ = self._make_familiar()
        topic_buffer.delete_if_idle.return_value = True

        result = await familiar.evict_topic(_identity_scope(), "topic_to_evict")

        assert result.topic_id == "topic_to_evict"
        assert result.removed is True
        topic_buffer.delete_if_idle.assert_called_once_with(
            _identity_scope(), "topic_to_evict"
        )

    @pytest.mark.asyncio
    async def test_scan_idle_buffers_once_skips_non_idle_topics(self):
        """验证空闲扫描跳过非空闲话题（服务候选为空）"""
        familiar, topic_buffer, _ = self._make_familiar(idle_timeout=3600)
        topic_buffer.list_idle_candidates.return_value = []

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == []
        topic_buffer.begin_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_idle_buffers_once_settles_idle_candidate(self):
        from datetime import datetime

        from hivememory.patchouli.services.topic_buffer import TopicCandidate

        familiar, topic_buffer, bus = self._make_familiar(idle_timeout=60)
        candidate = TopicCandidate(
            identity_scope=_identity_scope(),
            topic_id="stale_topic",
            state=__import__("hivememory").core.models.BufferState.IDLE,
            last_update=datetime.now().timestamp() - 3600,
            block_count=2,
        )
        topic_buffer.list_idle_candidates.return_value = [candidate]
        topic_buffer.begin_settlement.return_value = _reservation(
            "stale_topic", _task("stale_topic", TriggerReason.IDLE_TIMEOUT)
        )
        accepted = _generation_task("task-9", "stale_topic")
        bus.request = AsyncMock(return_value=accepted)
        topic_buffer.complete_settlement.return_value = _accepted_outcome(
            "stale_topic", task_id="task-9"
        )

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == ["stale_topic"]
        assert topic_buffer.begin_settlement.call_args.args[2] is TriggerReason.IDLE_TIMEOUT

    @pytest.mark.asyncio
    async def test_scan_idle_admission_failure_skips_and_preserves_topic(self):
        """idle 维护 admission 失败：记录失败并等待下一轮，不向上传播。"""
        from datetime import datetime

        from hivememory.core.models import BufferState
        from hivememory.patchouli.services.topic_buffer import TopicCandidate

        familiar, topic_buffer, bus = self._make_familiar(idle_timeout=60)
        candidate = TopicCandidate(
            identity_scope=_identity_scope(),
            topic_id="stale_topic",
            state=BufferState.IDLE,
            last_update=datetime.now().timestamp() - 3600,
            block_count=1,
        )
        topic_buffer.list_idle_candidates.return_value = [candidate]
        topic_buffer.begin_settlement.return_value = _reservation(
            "stale_topic", _task("stale_topic", TriggerReason.IDLE_TIMEOUT)
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == []
        topic_buffer.abort_settlement.assert_called_once()
        topic_buffer.complete_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_flush_classifies_settled_skipped_and_failed(self):
        """shutdown 逐 Topic 隔离：admission 失败记录为 failed，其余正常清理。"""
        from hivememory.core.models import BufferState
        from hivememory.patchouli.services.topic_buffer import TopicCandidate

        familiar, topic_buffer, bus = self._make_familiar()
        ok = TopicCandidate(
            identity_scope=_identity_scope(), topic_id="t-ok",
            state=BufferState.IDLE, last_update=1.0, block_count=2,
        )
        skip = TopicCandidate(
            identity_scope=_identity_scope(), topic_id="t-skip",
            state=BufferState.IDLE, last_update=1.0, block_count=1,
        )
        bad = TopicCandidate(
            identity_scope=_identity_scope(), topic_id="t-bad",
            state=BufferState.IDLE, last_update=1.0, block_count=3,
        )
        topic_buffer.list_shutdown_candidates.return_value = [ok, skip, bad]
        topic_buffer.begin_settlement.side_effect = [
            _reservation("t-ok", _task("t-ok", TriggerReason.SHUTDOWN)),
            _reservation("t-skip", None),  # 无材料
            _reservation("t-bad", _task("t-bad", TriggerReason.SHUTDOWN)),
        ]
        bus.request = AsyncMock(
            side_effect=[_generation_task("task-1", "t-ok"), RuntimeError("admission boom")]
        )
        topic_buffer.complete_settlement.side_effect = [
            _accepted_outcome("t-ok", task_id="task-1"),
            _accepted_outcome("t-skip"),
        ]

        report = await familiar.flush_all_for_shutdown()

        assert report.settled_topic_ids == ("t-ok", "t-skip")
        assert report.generation_skipped_topic_ids == ("t-skip",)
        assert report.failed_topic_ids == ("t-bad",)
        assert report.resident_block_count == 6
        topic_buffer.abort_settlement.assert_called_once()
