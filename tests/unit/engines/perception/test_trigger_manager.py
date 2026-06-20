"""
TriggerManager 单元测试

测试覆盖:
- DECISION_MATRIX 决策矩阵
- resolve_topic 调度器（使用 topic_id）
- 三种原子操作 (Archive/Compact/Evict)
- 依赖注入

Note:
    Phase 4.5 重构：resolve_topic 从 Identity 改为 topic_id
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from hivememory.core.models import Identity, TurnRecord
from hivememory.engines.perception.trigger_manager import TriggerManager, DECISION_MATRIX
from hivememory.engines.perception.models import (
    FlushReason,
    LogicalBlock,
)
from hivememory.patchouli.memory_library.models import TopicData


class TestDecisionMatrix:
    """DECISION_MATRIX 决策矩阵测试"""

    def test_token_overflow_actions(self):
        """TOKEN_OVERFLOW: Compact only"""
        actions = DECISION_MATRIX[FlushReason.TOKEN_OVERFLOW]
        assert actions["archive"] is False
        assert actions["compact"] is True
        assert actions["evict"] is False

    def test_idle_timeout_actions(self):
        """IDLE_TIMEOUT: Archive + Evict"""
        actions = DECISION_MATRIX[FlushReason.IDLE_TIMEOUT]
        assert actions["archive"] is True
        assert actions["compact"] is False
        assert actions["evict"] is True

    def test_lru_eviction_actions(self):
        """LRU_EVICTION: Archive + Evict"""
        actions = DECISION_MATRIX[FlushReason.LRU_EVICTION]
        assert actions["archive"] is True
        assert actions["compact"] is False
        assert actions["evict"] is True

    def test_manual_actions(self):
        """MANUAL: Archive + Compact"""
        actions = DECISION_MATRIX[FlushReason.MANUAL]
        assert actions["archive"] is True
        assert actions["compact"] is True
        assert actions["evict"] is False

    def test_shutdown_actions(self):
        """SHUTDOWN: Archive + Evict"""
        actions = DECISION_MATRIX[FlushReason.SHUTDOWN]
        assert actions["archive"] is True
        assert actions["compact"] is False
        assert actions["evict"] is True


class TestTriggerManagerInit:
    """TriggerManager 初始化测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_short_term_store = Mock()
        self.mock_relay_controller = Mock()

    def test_init_with_relay_controller(self):
        """测试带 RelayController 初始化"""
        manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=self.mock_relay_controller,
        )
        assert manager._store is self.mock_short_term_store
        assert manager._relay_controller is self.mock_relay_controller

    def test_init_requires_relay_controller(self):
        """TriggerManager requires explicit RelayController injection."""
        with pytest.raises(TypeError):
            TriggerManager(store=self.mock_short_term_store)


class TestTriggerManagerDependencyInjection:
    """TriggerManager 依赖注入测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_short_term_store = Mock()
        self.manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=Mock(),
        )

    def test_set_generation_callback(self):
        """测试注入 generation callback"""
        callback = AsyncMock(return_value=None)
        self.manager.set_generation_callback(callback)
        assert self.manager._on_generate_memory is callback


class TestTriggerManagerResolveTopic:
    """TriggerManager resolve_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_short_term_store = Mock()
        self.mock_relay_controller = Mock()
        self.mock_callback = AsyncMock(return_value=None)

        self.manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=self.mock_relay_controller,
        )
        self.manager.set_generation_callback(self.mock_callback)

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    def _create_topic_data_with_blocks(self, block_count: int = 3) -> TopicData:
        """辅助方法：创建带有 blocks 的只读话题数据"""
        blocks = []
        for i in range(block_count):
            blocks.append(
                LogicalBlock(
                    turn=TurnRecord(
                        user_query=f"Query {i}",
                        assistant_final_text=f"Response {i}",
                    ),
                    total_tokens=100,
                )
            )
        return TopicData(
            topic_id=self.topic_id,
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_title="测试话题",
            blocks=tuple(blocks),
            last_update=1.0,
            last_accessed_at=1.0,
            total_tokens=block_count * 100,
        )

    @pytest.mark.asyncio
    async def test_resolve_topic_empty_topic_data(self):
        """测试空 topic_data 跳过结算"""
        self.mock_short_term_store.get_topic_data.return_value = None

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)

        # 不应该调用任何依赖
        self.mock_callback.assert_not_called()
        self.mock_relay_controller.generate_summary.assert_not_called()
        self.mock_short_term_store.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_topic_topic_data_no_blocks(self):
        """测试无 blocks 的 topic_data 跳过结算"""
        topic_data = TopicData(
            topic_id=self.topic_id,
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_title="测试话题",
            last_update=1.0,
            last_accessed_at=1.0,
        )
        self.mock_short_term_store.get_topic_data.return_value = topic_data

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)

        # 不应该调用任何依赖
        self.mock_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_topic_idle_timeout(self):
        """测试 IDLE_TIMEOUT 触发 Archive + Evict"""
        topic_data = self._create_topic_data_with_blocks()
        self.mock_short_term_store.get_topic_data.return_value = topic_data

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args
        assert call_args[0][0].user_id == self.identity.user_id

        # 验证 Evict 被调用
        self.mock_short_term_store.pop_buffer.assert_called_once_with(self.topic_id)

        # 验证旧 blocks 通过 Store 命名方法清空
        self.mock_short_term_store.clear_blocks.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_resolve_topic_token_overflow(self):
        """测试 TOKEN_OVERFLOW 触发 Compact"""
        topic_data = self._create_topic_data_with_blocks()
        self.mock_short_term_store.get_topic_data.return_value = topic_data
        self.mock_relay_controller.generate_summary.return_value = "Test summary"

        await self.manager.resolve_topic(self.topic_id, FlushReason.TOKEN_OVERFLOW)

        # 验证 Compact 被调用
        self.mock_relay_controller.generate_summary.assert_called_once()

        # 验证 Archive 未被调用
        self.mock_callback.assert_not_called()

        # 验证 Evict 未被调用
        self.mock_short_term_store.pop_buffer.assert_not_called()

        # 验证摘要与旧 blocks 都通过 Store 命名方法写入
        self.mock_short_term_store.update_summary.assert_called_once_with(self.topic_id, "Test summary")
        self.mock_short_term_store.clear_blocks.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_resolve_topic_manual(self):
        """测试 MANUAL 触发 Archive + Compact"""
        topic_data = self._create_topic_data_with_blocks()
        self.mock_short_term_store.get_topic_data.return_value = topic_data
        self.mock_relay_controller.generate_summary.return_value = "Test summary"

        await self.manager.resolve_topic(self.topic_id, FlushReason.MANUAL)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()

        # 验证 Compact 被调用
        self.mock_relay_controller.generate_summary.assert_called_once()

        # 验证旧 blocks 通过 Store 命名方法清空
        self.mock_short_term_store.clear_blocks.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_resolve_topic_lru_eviction(self):
        """测试 LRU_EVICTION 触发 Archive + Evict"""
        topic_data = self._create_topic_data_with_blocks()
        self.mock_short_term_store.get_topic_data.return_value = topic_data

        await self.manager.resolve_topic(self.topic_id, FlushReason.LRU_EVICTION)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()

        # 验证 Evict 被调用
        self.mock_short_term_store.pop_buffer.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_resolve_topic_shutdown_waits_for_archive(self):
        """测试 SHUTDOWN 触发时等待 Archive 完成后再驱逐"""
        topic_data = self._create_topic_data_with_blocks()
        self.mock_short_term_store.get_topic_data.return_value = topic_data

        await self.manager.resolve_topic(
            self.topic_id,
            FlushReason.SHUTDOWN,
            wait_for_archive=True,
        )

        self.mock_callback.assert_awaited_once()
        self.mock_short_term_store.pop_buffer.assert_called_once_with(self.topic_id)
        self.mock_short_term_store.clear_blocks.assert_called_once_with(self.topic_id)


class TestTriggerManagerArchiveTopic:
    """TriggerManager _archive_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_short_term_store = Mock()
        self.mock_callback = AsyncMock(return_value=None)

        self.manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=Mock(),
        )
        self.manager.set_generation_callback(self.mock_callback)

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    @pytest.mark.asyncio
    async def test_archive_without_callback(self):
        """测试无回调时跳过 Archive"""
        manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=Mock(),
        )

        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test", assistant_final_text="test"),
                total_tokens=10,
            )
        ]

        await manager._archive_topic(
            self.topic_id,
            blocks,
            "summary",
            reason=FlushReason.IDLE_TIMEOUT,
        )

        # 不应该抛出异常
        assert True

    @pytest.mark.asyncio
    async def test_archive_filters_worth_saving_false(self):
        """测试过滤 worth_saving=False 的 block"""
        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test1", assistant_final_text="test1"),
                worth_saving=True,
                total_tokens=10,
            ),
            LogicalBlock(
                turn=TurnRecord(user_query="test2", assistant_final_text="test2"),
                worth_saving=False,
                total_tokens=10,
            ),
            LogicalBlock(
                turn=TurnRecord(user_query="test3", assistant_final_text="test3"),
                worth_saving=None,
                total_tokens=10,
            ),
        ]

        await self.manager._archive_topic(
            self.topic_id,
            blocks,
            "summary",
            reason=FlushReason.IDLE_TIMEOUT,
        )
        await asyncio.sleep(0)

        # 验证只发射了 2 个 block (worth_saving=True 和 None)
        call_args = self.mock_callback.call_args
        emitted_blocks = call_args[0][0].blocks
        assert len(emitted_blocks) == 2

    @pytest.mark.asyncio
    async def test_archive_payload_contains_identity(self):
        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test", assistant_final_text="test"),
                total_tokens=10,
            )
        ]

        await self.manager._archive_topic(
            self.topic_id,
            blocks,
            "summary",
            reason=FlushReason.IDLE_TIMEOUT,
            user_id=self.identity.user_id,
        )
        await asyncio.sleep(0)

        call_args = self.mock_callback.call_args
        assert call_args[0][0].user_id == self.identity.user_id

    @pytest.mark.asyncio
    async def test_archive_skips_all_filtered(self):
        """测试所有 blocks 被过滤时跳过 Archive"""
        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test", assistant_final_text="test"),
                worth_saving=False,
                total_tokens=10,
            )
        ]

        await self.manager._archive_topic(self.topic_id, blocks, "summary", None)
        await asyncio.sleep(0)

        self.mock_callback.assert_not_called()


class TestTriggerManagerCompactTopic:
    """TriggerManager _compact_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_short_term_store = Mock()
        self.mock_relay_controller = Mock()

        self.manager = TriggerManager(
            store=self.mock_short_term_store,
            relay_controller=self.mock_relay_controller,
        )

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    @pytest.mark.asyncio
    async def test_compact_updates_state_summary(self):
        """测试 Compact 更新 state_summary"""
        self.mock_relay_controller.generate_summary.return_value = "New summary"

        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test", assistant_final_text="test"),
                total_tokens=10,
            )
        ]

        await self.manager._compact_topic(self.topic_id, blocks, "previous summary")

        # 验证 state_summary 通过 Store 命名方法更新
        self.mock_short_term_store.update_summary.assert_called_once_with(self.topic_id, "New summary")
        self.mock_relay_controller.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_always_generates_summary(self):
        """_compact_topic 不做存在性检查；存在性由 resolve_topic 保证。"""
        self.mock_relay_controller.generate_summary.return_value = "summary"

        blocks = [
            LogicalBlock(
                turn=TurnRecord(user_query="test", assistant_final_text="test"),
                total_tokens=10,
            )
        ]

        await self.manager._compact_topic(self.topic_id, blocks, "summary")

        # generate_summary 仍会被调用；update_summary 在 store 层做 no-op
        self.mock_relay_controller.generate_summary.assert_called_once()
