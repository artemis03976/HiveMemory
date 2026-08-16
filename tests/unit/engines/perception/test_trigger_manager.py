from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, TopicData, TurnRecord
from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    LogicalBlock,
    TopicMaterializeTask,
)
from hivememory.engines.perception.trigger_manager import DECISION_MATRIX, TriggerManager


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

    def test_manual_settles_and_compacts_without_evict(self):
        actions = DECISION_MATRIX[FlushReason.MANUAL]
        assert actions == {"settle": True, "compact": True, "evict": False}

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
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_title="Test topic",
            topic_summary="Topic summary",
            blocks=blocks,
            last_update=1.0,
            last_accessed_at=1.0,
            total_tokens=block_count * 100,
            state_summary="previous summary",
        )

    @pytest.mark.asyncio
    async def test_empty_topic_data_returns_none(self):
        self.store.get_topic_data.return_value = None

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_topic_data_without_blocks_returns_none(self):
        self.store.get_topic_data.return_value = TopicData(
            topic_id=self.topic_id,
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_title="Empty topic",
            last_update=1.0,
            last_accessed_at=1.0,
        )

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_compacts_and_returns_no_settlement(self):
        self.store.get_topic_data.return_value = self._topic_data()
        self.relay.generate_summary.return_value = "new summary"
        self.store.apply_compaction.return_value = 1

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=2,
        )

        assert result is None
        folded_blocks = self.relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in folded_blocks] == ["Query 0"]
        self.store.apply_compaction.assert_called_once_with(
            self.topic_id,
            "new summary",
            retain_count=2,
        )
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_requires_explicit_retention_policy(self):
        self.store.get_topic_data.return_value = self._topic_data()

        with pytest.raises(ValueError, match="requires retain_recent_blocks"):
            await self.manager.resolve_topic(
                FlushEvent(
                    topic_id=self.topic_id,
                    reason=FlushReason.TOKEN_OVERFLOW,
                )
            )

        self.relay.generate_summary.assert_not_called()
        self.store.update_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_overflow_defers_when_all_blocks_are_retained(self):
        self.store.get_topic_data.return_value = self._topic_data(block_count=2)

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=3,
        )

        assert result is None
        self.relay.generate_summary.assert_not_called()
        self.store.update_summary.assert_not_called()
        self.store.clear_blocks.assert_not_called()
        self.store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_timeout_returns_settlement_and_evicts(self):
        self.store.get_topic_data.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.IDLE_TIMEOUT)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.topic_id == self.topic_id
        assert result.reason == FlushReason.IDLE_TIMEOUT
        assert len(result.blocks) == 3
        self.relay.generate_summary.assert_not_called()
        self.store.clear_blocks.assert_called_once_with(self.topic_id)
        self.store.pop_buffer.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_lru_eviction_returns_settlement_and_evicts(self):
        self.store.get_topic_data.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.LRU_EVICTION)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.reason == FlushReason.LRU_EVICTION
        self.store.clear_blocks.assert_called_once_with(self.topic_id)
        self.store.pop_buffer.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_shutdown_returns_settlement_and_evicts(self):
        self.store.get_topic_data.return_value = self._topic_data()

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.SHUTDOWN)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.reason == FlushReason.SHUTDOWN
        self.store.clear_blocks.assert_called_once_with(self.topic_id)
        self.store.pop_buffer.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_manual_returns_settlement_compacts_and_keeps_topic(self):
        self.store.get_topic_data.return_value = self._topic_data()
        self.relay.generate_summary.return_value = "manual summary"

        result = await self.manager.resolve_topic(
            FlushEvent(topic_id=self.topic_id, reason=FlushReason.MANUAL)
        )

        assert isinstance(result, TopicMaterializeTask)
        assert result.reason == FlushReason.MANUAL
        self.relay.generate_summary.assert_called_once()
        self.store.update_summary.assert_called_once_with(self.topic_id, "manual summary")
        self.store.clear_blocks.assert_called_once_with(self.topic_id)
        self.store.pop_buffer.assert_not_called()


class TestTriggerManagerSettlePayload:
    def setup_method(self):
        self.manager = TriggerManager(store=Mock(), relay_controller=Mock())

    @pytest.mark.asyncio
    async def test_settlement_filters_worth_saving_false(self):
        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="keep", assistant_final_text="keep"),
                worth_saving=True,
            ),
            LogicalBlock(
                turn=TurnRecord(user_query="drop", assistant_final_text="drop"),
                worth_saving=False,
            ),
            LogicalBlock(
                turn=TurnRecord(user_query="default", assistant_final_text="default"),
                worth_saving=None,
            ),
        ]

        payload = self.manager._build_settle_payload(
            topic_id="topic_1",
            blocks_snapshot=blocks,
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
            user_id="u1",
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
                    turn=TurnRecord(user_query="drop", assistant_final_text="drop"),
                    worth_saving=False,
                )
            ],
            state_summary="summary",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        assert payload is None


class TestTriggerManagerCompactTopic:
    @pytest.mark.asyncio
    async def test_compact_updates_state_summary(self):
        store = Mock()
        relay = Mock()
        relay.generate_summary.return_value = "new summary"
        manager = TriggerManager(store=store, relay_controller=relay)
        blocks = [
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ]

        await manager._compact_topic("topic_1", blocks, "previous summary")

        relay.generate_summary.assert_called_once_with(
            blocks_to_fold=blocks,
            previous_summary="previous summary",
        )
        store.update_summary.assert_called_once_with("topic_1", "new summary")

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

        folded = await manager._compact_topic(
            "topic_1",
            blocks,
            "previous summary",
            retain_recent_blocks=2,
        )

        assert folded == 2
        summarized = relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in summarized] == ["q0", "q1"]
        store.apply_compaction.assert_called_once_with(
            "topic_1",
            "new summary",
            retain_count=2,
        )
