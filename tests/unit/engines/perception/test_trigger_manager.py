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

from hivememory.core.models import Identity
from hivememory.engines.perception.trigger_manager import TriggerManager, DECISION_MATRIX
from hivememory.engines.perception.models import (
    FlushReason,
    LogicalBlock,
    SemanticBuffer,
)


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

    def test_mtp_write_actions(self):
        """MTP_WRITE: Archive + Compact"""
        actions = DECISION_MATRIX[FlushReason.MTP_WRITE]
        assert actions["archive"] is True
        assert actions["compact"] is True
        assert actions["evict"] is False

    def test_shutdown_actions(self):
        """SHUTDOWN: Archive + Evict"""
        actions = DECISION_MATRIX[FlushReason.SHUTDOWN]
        assert actions["archive"] is True
        assert actions["compact"] is False
        assert actions["evict"] is True

    def test_mtp_update_actions(self):
        """MTP_UPDATE: Archive + Compact"""
        actions = DECISION_MATRIX[FlushReason.MTP_UPDATE]
        assert actions["archive"] is True
        assert actions["compact"] is True
        assert actions["evict"] is False


class TestTriggerManagerInit:
    """TriggerManager 初始化测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_buffer_manager = Mock()
        self.mock_relay_controller = Mock()

    def test_init_with_relay_controller(self):
        """测试带 RelayController 初始化"""
        manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
            relay_controller=self.mock_relay_controller,
        )
        assert manager._buffer_manager is self.mock_buffer_manager
        assert manager._relay_controller is self.mock_relay_controller

    def test_init_without_relay_controller(self):
        """测试不带 RelayController 初始化"""
        manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
        )
        assert manager._buffer_manager is self.mock_buffer_manager
        assert manager._relay_controller is None


class TestTriggerManagerDependencyInjection:
    """TriggerManager 依赖注入测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_buffer_manager = Mock()
        self.manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
        )

    def test_set_generation_callback(self):
        """测试注入 generation callback"""
        callback = AsyncMock(return_value=None)
        self.manager.set_generation_callback(callback)
        assert self.manager._on_generate_memory is callback

    def test_set_relay_controller(self):
        """测试注入 RelayController"""
        mock_relay = Mock()
        self.manager.set_relay_controller(mock_relay)
        assert self.manager._relay_controller is mock_relay


class TestTriggerManagerResolveTopic:
    """TriggerManager resolve_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_buffer_manager = Mock()
        self.mock_relay_controller = Mock()
        self.mock_callback = AsyncMock(return_value=None)

        self.manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
            relay_controller=self.mock_relay_controller,
        )
        self.manager.set_generation_callback(self.mock_callback)

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    def _create_buffer_with_blocks(self, block_count: int = 3) -> SemanticBuffer:
        """辅助方法：创建带有 blocks 的 buffer"""
        buffer = SemanticBuffer(
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_id=self.topic_id,
        )
        for i in range(block_count):
            block = LogicalBlock(
                user_query=f"Query {i}",
                assistant_final_text=f"Response {i}",
                total_tokens=100,
            )
            buffer.blocks.append(block)
        buffer.total_tokens = block_count * 100
        return buffer

    @pytest.mark.asyncio
    async def test_resolve_topic_empty_buffer(self):
        """测试空 buffer 跳过结算"""
        self.mock_buffer_manager.get_buffer.return_value = None

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)

        # 不应该调用任何依赖
        self.mock_callback.assert_not_called()
        self.mock_relay_controller.generate_summary.assert_not_called()
        self.mock_buffer_manager.pop_buffer.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_topic_buffer_no_blocks(self):
        """测试无 blocks 的 buffer 跳过结算"""
        buffer = SemanticBuffer(
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_id=self.topic_id,
        )
        self.mock_buffer_manager.get_buffer.return_value = buffer

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)

        # 不应该调用任何依赖
        self.mock_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_topic_idle_timeout(self):
        """测试 IDLE_TIMEOUT 触发 Archive + Evict"""
        buffer = self._create_buffer_with_blocks()
        self.mock_buffer_manager.get_buffer.return_value = buffer

        await self.manager.resolve_topic(self.topic_id, FlushReason.IDLE_TIMEOUT)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args
        assert call_args[0][0].user_id == self.identity.user_id

        # 验证 Evict 被调用
        self.mock_buffer_manager.pop_buffer.assert_called_once_with(self.topic_id)

        # 验证 buffer 被清空
        assert len(buffer.blocks) == 0
        assert buffer.total_tokens == 0

    @pytest.mark.asyncio
    async def test_resolve_topic_token_overflow(self):
        """测试 TOKEN_OVERFLOW 触发 Compact"""
        buffer = self._create_buffer_with_blocks()
        self.mock_buffer_manager.get_buffer.return_value = buffer
        self.mock_relay_controller.generate_summary.return_value = "Test summary"

        await self.manager.resolve_topic(self.topic_id, FlushReason.TOKEN_OVERFLOW)

        # 验证 Compact 被调用
        self.mock_relay_controller.generate_summary.assert_called_once()

        # 验证 Archive 未被调用
        self.mock_callback.assert_not_called()

        # 验证 Evict 未被调用
        self.mock_buffer_manager.pop_buffer.assert_not_called()

        # 验证 buffer 状态
        assert buffer.state_summary == "Test summary"
        assert len(buffer.blocks) == 0
        assert buffer.total_tokens == 0

    @pytest.mark.asyncio
    async def test_resolve_topic_mtp_write(self):
        """测试 MTP_WRITE 触发 Archive + Compact"""
        buffer = self._create_buffer_with_blocks()
        self.mock_buffer_manager.get_buffer.return_value = buffer
        self.mock_relay_controller.generate_summary.return_value = "Test summary"

        await self.manager.resolve_topic(self.topic_id, FlushReason.MTP_WRITE)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()

        # 验证 Compact 被调用
        self.mock_relay_controller.generate_summary.assert_called_once()

        # 验证 buffer 被清空
        assert len(buffer.blocks) == 0

    @pytest.mark.asyncio
    async def test_resolve_topic_lru_eviction(self):
        """测试 LRU_EVICTION 触发 Archive + Evict"""
        buffer = self._create_buffer_with_blocks()
        self.mock_buffer_manager.get_buffer.return_value = buffer

        await self.manager.resolve_topic(self.topic_id, FlushReason.LRU_EVICTION)
        await asyncio.sleep(0)

        # 验证 Archive 被调用
        self.mock_callback.assert_called_once()

        # 验证 Evict 被调用
        self.mock_buffer_manager.pop_buffer.assert_called_once_with(self.topic_id)

    @pytest.mark.asyncio
    async def test_resolve_topic_shutdown_waits_for_archive(self):
        """测试 SHUTDOWN 触发时等待 Archive 完成后再驱逐"""
        buffer = self._create_buffer_with_blocks()
        self.mock_buffer_manager.get_buffer.return_value = buffer

        await self.manager.resolve_topic(
            self.topic_id,
            FlushReason.SHUTDOWN,
            wait_for_archive=True,
        )

        self.mock_callback.assert_awaited_once()
        self.mock_buffer_manager.pop_buffer.assert_called_once_with(self.topic_id)
        assert len(buffer.blocks) == 0
        assert buffer.total_tokens == 0


class TestTriggerManagerArchiveTopic:
    """TriggerManager _archive_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_buffer_manager = Mock()
        self.mock_callback = AsyncMock(return_value=None)

        self.manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
        )
        self.manager.set_generation_callback(self.mock_callback)

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    @pytest.mark.asyncio
    async def test_archive_without_callback(self):
        """测试无回调时跳过 Archive"""
        manager = TriggerManager(buffer_manager=self.mock_buffer_manager)

        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", total_tokens=10)
        ]

        await manager._archive_topic(self.topic_id, blocks, "summary", None)

        # 不应该抛出异常
        assert True

    @pytest.mark.asyncio
    async def test_archive_filters_worth_saving_false(self):
        """测试过滤 worth_saving=False 的 block"""
        blocks = [
            LogicalBlock(user_query="test1", assistant_final_text="test1", worth_saving=True, total_tokens=10),
            LogicalBlock(user_query="test2", assistant_final_text="test2", worth_saving=False, total_tokens=10),
            LogicalBlock(user_query="test3", assistant_final_text="test3", worth_saving=None, total_tokens=10),
        ]

        await self.manager._archive_topic(self.topic_id, blocks, "summary", None)
        await asyncio.sleep(0)

        # 验证只发射了 2 个 block (worth_saving=True 和 None)
        call_args = self.mock_callback.call_args
        emitted_blocks = call_args[0][0].blocks
        assert len(emitted_blocks) == 2

    @pytest.mark.asyncio
    async def test_archive_payload_contains_identity(self):
        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", total_tokens=10)
        ]

        await self.manager._archive_topic(
            self.topic_id, blocks, "summary", None, FlushReason.IDLE_TIMEOUT, self.identity.user_id
        )
        await asyncio.sleep(0)

        call_args = self.mock_callback.call_args
        assert call_args[0][0].user_id == self.identity.user_id

    @pytest.mark.asyncio
    async def test_archive_skips_all_filtered(self):
        """测试所有 blocks 被过滤时跳过 Archive"""
        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", worth_saving=False, total_tokens=10)
        ]

        await self.manager._archive_topic(self.topic_id, blocks, "summary", None)
        await asyncio.sleep(0)

        self.mock_callback.assert_not_called()


class TestTriggerManagerCompactTopic:
    """TriggerManager _compact_topic 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.mock_buffer_manager = Mock()
        self.mock_relay_controller = Mock()

        self.manager = TriggerManager(
            buffer_manager=self.mock_buffer_manager,
            relay_controller=self.mock_relay_controller,
        )

        self.topic_id = "test_topic_123"
        self.identity = Identity(user_id="user1", agent_id="agent1")

    @pytest.mark.asyncio
    async def test_compact_updates_state_summary(self):
        """测试 Compact 更新 state_summary"""
        buffer = SemanticBuffer(
            user_id=self.identity.user_id,
            current_agent_id=self.identity.agent_id,
            topic_id=self.topic_id,
        )
        self.mock_buffer_manager.get_buffer.return_value = buffer
        self.mock_relay_controller.generate_summary.return_value = "New summary"

        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", total_tokens=10)
        ]

        await self.manager._compact_topic(self.topic_id, blocks, "previous summary")

        # 验证 state_summary 更新
        assert buffer.state_summary == "New summary"
        self.mock_relay_controller.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_without_relay_controller(self):
        """测试无 RelayController 时跳过 Compact"""
        manager = TriggerManager(buffer_manager=self.mock_buffer_manager)

        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", total_tokens=10)
        ]

        await manager._compact_topic(self.topic_id, blocks, "summary")

        # 不应该抛出异常
        self.mock_relay_controller.generate_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_without_buffer(self):
        """测试无 buffer 时跳过 Compact"""
        self.mock_buffer_manager.get_buffer.return_value = None

        blocks = [
            LogicalBlock(user_query="test", assistant_final_text="test", total_tokens=10)
        ]

        await self.manager._compact_topic(self.topic_id, blocks, "summary")

        self.mock_relay_controller.generate_summary.assert_not_called()
