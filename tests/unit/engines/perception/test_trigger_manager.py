from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from hivememory.core.models import Identity, TopicData, TurnRecord, WorkspaceTopicKey
from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    LogicalBlock,
    TopicMaterializeTask,
)
from hivememory.engines.perception.trigger_manager import DECISION_MATRIX, TriggerManager
from hivememory.patchouli.errors import TopicBusyError
from tests.helpers.workspace import make_identity_scope


class TestDecisionMatrix:
    def test_token_overflow_compacts_only(self):
        actions = DECISION_MATRIX[FlushReason.TOKEN_OVERFLOW]
        assert actions == {"settle": False, "compact": True, "evict": False}

    def test_idle_timeout_settles_and_evicts(self):
        actions = DECISION_MATRIX[FlushReason.IDLE_TIMEOUT]
        assert actions == {"settle": True, "compact": False, "evict": True}

    def test_lru_eviction_settles_and_evicts(self):
        actions = DECISION_MATRIX[FlushReason.LRU_EVICTION]
        assert actions == {"settle": True, "compact": False, "evict": True}

    def test_shutdown_settles_and_evicts(self):
        actions = DECISION_MATRIX[FlushReason.SHUTDOWN]
        assert actions == {"settle": True, "compact": False, "evict": True}

    def test_manual_settle_settles_and_evicts_without_compact(self):
        actions = DECISION_MATRIX[FlushReason.MANUAL_SETTLE]
        assert actions == {"settle": True, "compact": False, "evict": True}

    def test_manual_compact_compacts_only(self):
        actions = DECISION_MATRIX[FlushReason.MANUAL_COMPACT]
        assert actions == {"settle": False, "compact": True, "evict": False}

    def test_manual_delete_evicts_only(self):
        actions = DECISION_MATRIX[FlushReason.MANUAL_DELETE]
        assert actions == {"settle": False, "compact": False, "evict": True}

    def test_legacy_manual_reason_is_removed(self):
        assert "MANUAL" not in {reason.name for reason in FlushReason}
        assert "MANUAL" not in {reason.name for reason in DECISION_MATRIX}

    def test_active_write_reasons_are_not_perception_flush_reasons(self):
        reason_names = {reason.name for reason in FlushReason}
        assert "MTP_WRITE" not in reason_names
        assert "MTP_UPDATE" not in reason_names
        assert "MTP_WRITE" not in {reason.name for reason in DECISION_MATRIX}
        assert "MTP_UPDATE" not in {reason.name for reason in DECISION_MATRIX}


class TestTriggerManagerInit:
    def test_init_requires_relay_controller(self):
        with pytest.raises(TypeError):
            TriggerManager(store=Mock())


class TestTriggerManagerResolveTopic:
    def setup_method(self):
        self.store = Mock()
        self.relay = Mock()
        self.manager = TriggerManager(store=self.store, relay_controller=self.relay)
        self.topic_id = "topic_1"
        self.identity = Identity(user_id="user1", agent_id="agent1")
        self.identity_scope = make_identity_scope(actor_identity=self.identity)
        self.topic_key = WorkspaceTopicKey.from_identity_scope(
            self.identity_scope, self.topic_id
        )

    def _topic_data(self, block_count: int = 3) -> TopicData:
        blocks = tuple(
            LogicalBlock(
                turn=TurnRecord(
                    identity=self.identity,
                    user_query=f"Query {i}",
                    assistant_final_text=f"Response {i}",
                ),
                total_tokens=100,
            )
            for i in range(block_count)
        )
        return TopicData(
            topic_id=self.topic_id,
            workspace_identity=self.identity_scope.workspace_identity,
            current_agent_id=self.identity.agent_id,
            topic_title="Test topic",
            topic_summary="Topic summary",
            blocks=blocks,
            last_update=1.0,
            last_accessed_at=1.0,
            total_tokens=block_count * 100,
            state_summary="previous summary",
        )

    def test_automatic_settle_reports_missing_without_eviction(self):
        """目标缺失与已驱逐空 Topic 必须是两个不同结果。"""
        self.store.freeze_and_evict.return_value = None

        result = self.manager.settle_and_evict(
            self.topic_key,
            FlushReason.IDLE_TIMEOUT,
        )

        assert result.evicted is False
        assert result.settlement is None

    def test_automatic_settle_reports_empty_topic_as_evicted_without_payload(self):
        """空 Topic 已完成生命周期，只是没有 generation 材料。"""
        self.store.freeze_and_evict.return_value = TopicData(
            topic_id=self.topic_id,
            workspace_identity=self.identity_scope.workspace_identity,
            current_agent_id=self.identity.agent_id,
            topic_title="Empty topic",
            last_update=1.0,
            last_accessed_at=1.0,
        )

        result = self.manager.settle_and_evict(
            self.topic_key,
            FlushReason.IDLE_TIMEOUT,
        )

        assert result.evicted is True
        assert result.settlement is None

    @pytest.mark.asyncio
    async def test_empty_topic_data_returns_none(self):
        self.store.get_topic_data_by_key.return_value = None

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()
        self.store.pop_buffer_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_truly_empty_topic_is_evicted_when_matrix_says_evict(self):
        """真正空 Topic 没有 settle/compact 材料，但仍需按矩阵执行 evict。"""
        self.store.freeze_and_evict.return_value = TopicData(
            topic_id=self.topic_id,
            workspace_identity=self.identity_scope.workspace_identity,
            current_agent_id=self.identity.agent_id,
            topic_title="Empty topic",
            last_update=1.0,
            last_accessed_at=1.0,
        )

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.freeze_and_evict.assert_called_once_with(self.topic_key)
        self.store.pop_buffer_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_truly_empty_topic_without_evict_stays_untouched(self):
        """TOKEN_OVERFLOW（evict=False）下空 Topic 不应被修改。"""
        self.store.get_topic_data_by_key.return_value = TopicData(
            topic_id=self.topic_id,
            workspace_identity=self.identity_scope.workspace_identity,
            current_agent_id=self.identity.agent_id,
            topic_title="Empty topic",
            last_update=1.0,
            last_accessed_at=1.0,
        )

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=1,
        )

        assert result is None
        self.store.pop_buffer_by_key.assert_not_called()
        self.store.apply_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_compacts_and_returns_no_settlement(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()
        self.relay.generate_summary.return_value = "new summary"
        self.store.apply_compaction.return_value = 1

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=2,
        )

        assert result is None
        folded_blocks = self.relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in folded_blocks] == ["Query 0"]
        self.store.apply_compaction.assert_called_once_with(
            self.topic_key,
            "new summary",
            retain_count=2,
        )
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_requires_explicit_retention_policy(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()

        with pytest.raises(ValueError, match="requires retain_recent_blocks"):
            await self.manager.resolve_topic(
                FlushEvent(
                    topic_key=self.topic_key,
                    reason=FlushReason.TOKEN_OVERFLOW,
                )
            )

        self.relay.generate_summary.assert_not_called()
        self.store.update_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_defers_when_all_blocks_are_retained(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data(block_count=2)

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=3,
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.update_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_timeout_returns_settlement_and_evicts(self):
        self.store.freeze_and_evict.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.topic_id == self.topic_id
        assert result.reason == FlushReason.IDLE_TIMEOUT
        assert len(result.blocks) == 3
        self.relay.generate_summary.assert_not_called()
        self.store.freeze_and_evict.assert_called_once_with(self.topic_key)

    @pytest.mark.asyncio
    async def test_lru_eviction_returns_settlement_and_evicts(self):
        self.store.freeze_and_evict.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.LRU_EVICTION)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.reason == FlushReason.LRU_EVICTION
        self.store.freeze_and_evict.assert_called_once_with(self.topic_key)

    @pytest.mark.asyncio
    async def test_shutdown_returns_settlement_and_evicts(self):
        self.store.freeze_and_evict.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.SHUTDOWN)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.reason == FlushReason.SHUTDOWN
        self.store.freeze_and_evict.assert_called_once_with(self.topic_key)

    @pytest.mark.asyncio
    async def test_manual_settle_is_not_dispatched_through_resolve_topic(self):
        """manual settle 必须走 FLUSHING prepare，不能经 resolve_topic 冻结驱逐。"""
        with pytest.raises(ValueError, match="prepare_manual_settle"):
            await self.manager.resolve_topic(
                FlushEvent(topic_key=self.topic_key, reason=FlushReason.MANUAL_SETTLE)
            )

    @pytest.mark.asyncio
    async def test_manual_compact_compacts_only_and_keeps_topic(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()
        self.relay.generate_summary.return_value = "manual summary"

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.MANUAL_COMPACT),
            retain_recent_blocks=1,
        )

        assert result is None
        summarized = self.relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in summarized] == ["Query 0", "Query 1"]
        self.store.apply_compaction.assert_called_once_with(
            self.topic_key,
            "manual summary",
            retain_count=1,
        )
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_compact_requires_retention_policy(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()

        with pytest.raises(ValueError, match="requires retain_recent_blocks"):
            await self.manager.resolve_topic(
                FlushEvent(
                    topic_key=self.topic_key,
                    reason=FlushReason.MANUAL_COMPACT,
                )
            )

        self.relay.generate_summary.assert_not_called()
        self.store.apply_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_compact_rejects_retain_below_one(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()

        with pytest.raises(ValueError, match="retain_recent_blocks must be >= 1"):
            await self.manager.resolve_topic(
                FlushEvent(
                    topic_key=self.topic_key,
                    reason=FlushReason.MANUAL_COMPACT,
                ),
                retain_recent_blocks=0,
            )

        self.relay.generate_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_delete_evicts_without_settlement_or_compact(self):
        self.store.get_topic_data_by_key.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_key=self.topic_key, reason=FlushReason.MANUAL_DELETE)
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer_by_key.assert_called_once_with(self.topic_key)

    @pytest.mark.asyncio
    async def test_prepare_manual_settle_is_non_destructive(self):
        """manual settle 的 prepare 阶段只冻结材料，不修改 buffer。"""
        self.store.freeze_for_manual_settle.return_value = self._topic_data()

        payload = self.manager.prepare_manual_settle(self.topic_key)

        assert isinstance(payload, TopicMaterializeTask)
        assert len(payload.blocks) == 3
        assert payload.reason == FlushReason.MANUAL_SETTLE
        self.store.freeze_for_manual_settle.assert_called_once_with(self.topic_key)
        self.store.pop_buffer_by_key.assert_not_called()
        self.store.apply_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_manual_settle_returns_none_for_empty_topic(self):
        self.store.freeze_for_manual_settle.return_value = TopicData(
            topic_id=self.topic_id,
            workspace_identity=self.identity_scope.workspace_identity,
            current_agent_id=self.identity.agent_id,
            topic_title="Empty topic",
            last_update=1.0,
            last_accessed_at=1.0,
        )

        payload = self.manager.prepare_manual_settle(self.topic_key)

        assert payload is None
        self.store.pop_buffer_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_manual_settle_raises_on_busy_topic(self):
        """busy（非 IDLE）Topic 无法取得 FLUSHING，必须显式报 busy。"""
        self.store.freeze_for_manual_settle.return_value = None

        with pytest.raises(TopicBusyError):
            self.manager.prepare_manual_settle(self.topic_key)


class TestTriggerManagerSettlePayload:
    def setup_method(self):
        self.manager = TriggerManager(store=Mock(), relay_controller=Mock())
        self.workspace_identity = make_identity_scope(user_id="u1").workspace_identity

    @pytest.mark.asyncio
    async def test_settlement_filters_worth_saving_false(self):
        blocks = [
            LogicalBlock(
                turn=TurnRecord(identity=Identity(user_id="u1"), user_query="keep", assistant_final_text="keep"),
                worth_saving=True,
            ),
            LogicalBlock(
                turn=TurnRecord(identity=Identity(user_id="u1"), user_query="drop", assistant_final_text="drop"),
                worth_saving=False,
            ),
            LogicalBlock(
                turn=TurnRecord(identity=Identity(user_id="u1"), user_query="default", assistant_final_text="default"),
                worth_saving=None,
            ),
        ]

        payload = self.manager._build_settle_payload(
            topic_id="topic_1",
            blocks_snapshot=blocks,
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
            workspace_identity=self.workspace_identity,
        )

        assert payload is not None
        assert len(payload.blocks) == 2
        assert [block.user_query for block in payload.blocks] == ["keep", "default"]

    @pytest.mark.asyncio
    async def test_settlement_returns_none_when_all_blocks_filtered(self):
        payload = self.manager._build_settle_payload(
            topic_id="topic_1",
            blocks_snapshot=[
                LogicalBlock(
                    turn=TurnRecord(identity=Identity(user_id="u1"), user_query="drop", assistant_final_text="drop"),
                    worth_saving=False,
                )
            ],
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
            workspace_identity=self.workspace_identity,
        )

        assert payload is None

    def test_settlement_task_rejects_field_reassignment(self):
        """进入 journal/queue 的 settlement 快照不能被调用方原地改写。"""
        payload = self.manager._build_settle_payload(
            topic_id="topic_1",
            blocks_snapshot=[
                LogicalBlock(
                    turn=TurnRecord(
                        identity=Identity(user_id="u1"),
                        user_query="keep",
                        assistant_final_text="answer",
                    )
                )
            ],
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
            workspace_identity=self.workspace_identity,
        )
        assert payload is not None

        with pytest.raises(ValidationError, match="frozen"):
            payload.topic_title = "mutated"

        assert payload.topic_title == ""

    def test_settlement_task_blocks_cannot_be_mutated_in_place(self):
        """冻结模型还必须冻结 blocks 容器，不能只禁止字段重新赋值。"""
        payload = self.manager._build_settle_payload(
            topic_id="topic_1",
            blocks_snapshot=[
                LogicalBlock(
                    turn=TurnRecord(
                        identity=Identity(user_id="u1"),
                        user_query="keep",
                        assistant_final_text="answer",
                    )
                )
            ],
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
            workspace_identity=self.workspace_identity,
        )
        assert payload is not None

        with pytest.raises(AttributeError, match="append"):
            payload.blocks.append(
                LogicalBlock(
                    turn=TurnRecord(
                        identity=Identity(user_id="u1"),
                        user_query="late",
                        assistant_final_text="mutation",
                    )
                )
            )

        assert [block.user_query for block in payload.blocks] == ["keep"]


class TestTriggerManagerCompactTopic:
    @pytest.mark.asyncio
    async def test_compact_updates_state_summary_and_trims_old_prefix(self):
        store = Mock()
        store.apply_compaction.return_value = 1
        relay = Mock()
        relay.generate_summary.return_value = "new summary"
        manager = TriggerManager(store=store, relay_controller=relay)
        blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q0", assistant_final_text="a0")),
            LogicalBlock(turn=TurnRecord(user_query="q1", assistant_final_text="a1")),
        ]

        topic_key = WorkspaceTopicKey.from_identity_scope(
            make_identity_scope(user_id="u1"), "topic_1"
        )
        folded = await manager._compact_topic(
            topic_key, blocks, "previous summary", retain_recent_blocks=1
        )

        assert folded == 1
        relay.generate_summary.assert_called_once_with(
            blocks_to_fold=[blocks[0]],
            previous_summary="previous summary",
        )
        store.apply_compaction.assert_called_once_with(
            topic_key,
            "new summary",
            retain_count=1,
        )

    @pytest.mark.asyncio
    async def test_compact_rejects_retain_below_one(self):
        store = Mock()
        relay = Mock()
        manager = TriggerManager(store=store, relay_controller=relay)
        topic_key = WorkspaceTopicKey.from_identity_scope(
            make_identity_scope(user_id="u1"), "topic_1"
        )

        for bad in (0, -1):
            with pytest.raises(ValueError, match="retain_recent_blocks must be >= 1"):
                await manager._compact_topic(
                    topic_key,
                    [LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))],
                    "previous summary",
                    retain_recent_blocks=bad,
                )
        relay.generate_summary.assert_not_called()
        store.apply_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_is_noop_when_blocks_not_exceeding_retain(self):
        store = Mock()
        relay = Mock()
        manager = TriggerManager(store=store, relay_controller=relay)
        blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ]

        topic_key = WorkspaceTopicKey.from_identity_scope(
            make_identity_scope(user_id="u1"), "topic_1"
        )
        folded = await manager._compact_topic(
            topic_key, blocks, "previous summary", retain_recent_blocks=2
        )

        assert folded == 0
        relay.generate_summary.assert_not_called()
        store.apply_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_summarizes_only_prefix_before_retained_blocks(self):
        store = Mock()
        store.apply_compaction.return_value = 2
        relay = Mock()
        relay.generate_summary.return_value = "new summary"
        manager = TriggerManager(store=store, relay_controller=relay)
        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query=f"q{i}", assistant_final_text=f"a{i}")
            )
            for i in range(4)
        ]

        topic_key = WorkspaceTopicKey.from_identity_scope(
            make_identity_scope(user_id="u1"), "topic_1"
        )
        folded = await manager._compact_topic(
            topic_key,
            blocks,
            "previous summary",
            retain_recent_blocks=2,
        )

        assert folded == 2
        summarized = relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in summarized] == ["q0", "q1"]
        store.apply_compaction.assert_called_once_with(
            topic_key,
            "new summary",
            retain_count=2,
        )
