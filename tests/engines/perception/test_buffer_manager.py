"""
BufferManager 单元测试

测试覆盖:
- SemanticBufferManager (原 BufferManager)
    - Buffer 获取和创建
    - Builder 获取和创建
    - Buffer CRUD 操作
    - Buffer 元数据更新
    - 多会话隔离
- SimpleBufferManager
    - Buffer 获取和创建
    - 消息添加
    - Buffer 清空

Note:
    v3.0 重构：BufferManager 分拆为 SemanticBufferManager 和 SimpleBufferManager
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from hivememory.core.models import Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.buffer_manager import (
    SemanticBufferManager,
    SimpleBufferManager
)
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
    SimpleBuffer,
)


class TestSemanticBufferManagerBasic:
    """SemanticBufferManager 基础功能测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )

    # ========== Buffer 获取和创建测试 ==========

    def test_get_buffer_creates_new(self):
        """测试获取不存在的 buffer 时创建新的"""
        buffer = self.manager.get_buffer(self.identity)

        assert buffer is not None
        assert buffer.identity.user_id == "user1"
        assert buffer.identity.agent_id == "agent1"
        assert buffer.identity.session_id == "session1"
        assert len(buffer.blocks) == 0

    def test_get_buffer_returns_existing(self):
        """测试获取已存在的 buffer"""
        buffer1 = self.manager.get_buffer(self.identity)
        buffer2 = self.manager.get_buffer(self.identity)

        assert buffer1 is buffer2

    # ========== Builder 获取和创建测试 ==========

    def test_get_builder_creates_new(self):
        """测试获取不存在的 builder 时创建新的"""
        builder = self.manager.get_builder(self.identity)

        assert builder is not None
        assert builder.is_empty

    def test_get_builder_returns_existing(self):
        """测试获取已存在的 builder"""
        builder1 = self.manager.get_builder(self.identity)
        builder2 = self.manager.get_builder(self.identity)

        assert builder1 is builder2

    def test_reset_builder(self):
        """测试重置 builder"""
        builder = self.manager.get_builder(self.identity)

        # 开始构建
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        builder.start(rewritten_query="Hello")
        builder.add_message(user_msg)

        assert builder.is_started

        # 重置
        self.manager.reset_builder(self.identity)

        # 验证重置后状态
        builder = self.manager.get_builder(self.identity)
        assert builder.is_empty


class TestSemanticBufferManagerCRUD:
    """SemanticBufferManager CRUD 操作测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )

    def _create_block(self, content: str = "Hello") -> LogicalBlock:
        """辅助方法：创建一个完整的 block"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content=content
        )
        response_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Response"
        )
        return LogicalBlock(
            user_block=user_msg,
            response_block=response_msg,
            total_tokens=100,
        )

    # ========== add_block_to_buffer 测试 ==========

    def test_add_block_to_buffer(self):
        """测试添加 block 到 buffer"""
        block = self._create_block()

        self.manager.add_block_to_buffer(self.identity, block)

        buffer = self.manager.get_buffer(self.identity)
        assert buffer is not None
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0] is block
        assert buffer.total_tokens == 100

    def test_add_multiple_blocks(self):
        """测试添加多个 blocks"""
        block1 = self._create_block("Hello 1")
        block2 = self._create_block("Hello 2")

        self.manager.add_block_to_buffer(self.identity, block1)
        self.manager.add_block_to_buffer(self.identity, block2)

        buffer = self.manager.get_buffer(self.identity)
        assert len(buffer.blocks) == 2
        assert buffer.total_tokens == 200

    # ========== clear_buffer 测试 ==========

    def test_clear_buffer_returns_blocks(self):
        """测试清空 buffer 返回被清除的 blocks"""
        block1 = self._create_block("Hello 1")
        block2 = self._create_block("Hello 2")

        self.manager.add_block_to_buffer(self.identity, block1)
        self.manager.add_block_to_buffer(self.identity, block2)

        cleared = self.manager.clear_buffer(self.identity)

        assert len(cleared) == 2
        assert block1 in cleared
        assert block2 in cleared

    def test_clear_buffer_empties_buffer(self):
        """测试清空 buffer 后 buffer 为空"""
        block = self._create_block()
        self.manager.add_block_to_buffer(self.identity, block)

        self.manager.clear_buffer(self.identity)

        buffer = self.manager.get_buffer(self.identity)
        assert len(buffer.blocks) == 0
        assert buffer.total_tokens == 0

    def test_clear_nonexistent_buffer_returns_empty(self):
        """测试清空不存在的 buffer 返回空列表"""
        cleared = self.manager.clear_buffer(self.identity)
        assert cleared == []

    # ========== update_buffer_metadata 测试 ==========

    def test_update_topic_kernel_vector(self):
        """测试更新话题核心向量"""
        self.manager.get_buffer(self.identity)

        new_vector = [0.1, 0.2, 0.3]
        self.manager.update_buffer_metadata(
            self.identity,
            topic_kernel_vector=new_vector
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.topic_kernel_vector == new_vector

    def test_reset_topic_kernel_vector(self):
        """测试重置话题核心向量"""
        self.manager.get_buffer(self.identity)
        self.manager.update_buffer_metadata(
            self.identity,
            topic_kernel_vector=[0.1, 0.2, 0.3]
        )

        self.manager.update_buffer_metadata(
            self.identity,
            reset_topic_kernel=True
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.topic_kernel_vector is None

    def test_update_relay_summary(self):
        """测试更新接力摘要"""
        self.manager.get_buffer(self.identity)

        self.manager.update_buffer_metadata(
            self.identity,
            relay_summary="Test summary"
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.relay_summary == "Test summary"

    def test_reset_relay_summary(self):
        """测试重置接力摘要"""
        self.manager.get_buffer(self.identity)
        self.manager.update_buffer_metadata(
            self.identity,
            relay_summary="Test summary"
        )

        self.manager.update_buffer_metadata(
            self.identity,
            reset_relay_summary=True
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.relay_summary is None

    def test_update_state(self):
        """测试更新状态"""
        self.manager.get_buffer(self.identity)

        self.manager.update_buffer_metadata(
            self.identity,
            state=BufferState.PROCESSING
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.state == BufferState.PROCESSING

    def test_update_multiple_fields(self):
        """测试同时更新多个字段"""
        self.manager.get_buffer(self.identity)

        self.manager.update_buffer_metadata(
            self.identity,
            topic_kernel_vector=[0.1, 0.2],
            relay_summary="Summary",
            state=BufferState.FLUSHING
        )

        buffer = self.manager.get_buffer(self.identity)
        assert buffer.topic_kernel_vector == [0.1, 0.2]
        assert buffer.relay_summary == "Summary"
        assert buffer.state == BufferState.FLUSHING


class TestSemanticBufferManagerMultiSession:
    """SemanticBufferManager 多会话测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()

    def test_multi_session_isolation(self):
        """测试多会话隔离"""
        identity1 = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )
        identity2 = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session2"
        )

        buffer1 = self.manager.get_buffer(identity1)
        buffer2 = self.manager.get_buffer(identity2)

        assert buffer1 is not buffer2
        assert buffer1.identity.session_id == "session1"
        assert buffer2.identity.session_id == "session2"

    def test_list_active_buffers(self):
        """测试列出活跃 buffers"""
        identity1 = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )
        identity2 = Identity(
            user_id="user2",
            agent_id="agent1",
            session_id="session1"
        )

        self.manager.get_buffer(identity1)
        self.manager.get_buffer(identity2)

        active_buffers = self.manager.list_active_buffers()

        assert len(active_buffers) == 2
        assert "user1:agent1:session1" in active_buffers
        assert "user2:agent1:session1" in active_buffers


class TestSemanticBufferManagerInfo:
    """SemanticBufferManager 信息查询测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )

    def test_get_buffer_info_nonexistent(self):
        """测试获取不存在 buffer 的信息"""
        info = self.manager.get_buffer_info(self.identity)
        assert info["exists"] is False

    def test_get_buffer_info_existing(self):
        """测试获取存在 buffer 的信息"""
        self.manager.get_buffer(self.identity)

        info = self.manager.get_buffer_info(self.identity)

        assert info["exists"] is True
        assert info["block_count"] == 0
        assert info["total_tokens"] == 0
        assert info["state"] == "idle"

    def test_get_buffer_info_with_blocks(self):
        """测试获取有 blocks 的 buffer 信息"""
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        response_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="Hi"
        )
        block = LogicalBlock(
            user_block=user_msg,
            response_block=response_msg,
            total_tokens=50,
        )

        self.manager.add_block_to_buffer(self.identity, block)

        info = self.manager.get_buffer_info(self.identity)

        assert info["exists"] is True
        assert info["block_count"] == 1
        assert info["total_tokens"] == 50

    def test_get_buffer_info_with_topic_kernel(self):
        """测试获取有话题核心的 buffer 信息"""
        self.manager.get_buffer(self.identity)
        self.manager.update_buffer_metadata(
            self.identity,
            topic_kernel_vector=[0.1, 0.2, 0.3]
        )

        info = self.manager.get_buffer_info(self.identity)

        assert info["has_topic_kernel"] is True

    def test_get_buffer_info_with_building_block(self):
        """测试获取有构建中 block 的 buffer 信息"""
        builder = self.manager.get_builder(self.identity)
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        builder.start(rewritten_query="Hello")
        builder.add_message(user_msg)

        # 确保 buffer 也存在
        self.manager.get_buffer(self.identity)

        info = self.manager.get_buffer_info(self.identity)

        assert info["exists"] is True
        assert info["has_building_block"] is True
        assert info["building_block_complete"] is False


class TestSimpleBufferManager:
    """SimpleBufferManager 测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SimpleBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
            session_id="session1"
        )

    def test_get_buffer_creates_new(self):
        """测试获取不存在的 buffer 时创建新的"""
        buffer = self.manager.get_buffer(self.identity)

        assert buffer is not None
        assert isinstance(buffer, SimpleBuffer)
        assert buffer.user_id == "user1"
        assert buffer.agent_id == "agent1"
        assert buffer.session_id == "session1"

    def test_get_buffer_returns_existing(self):
        """测试获取已存在的 buffer"""
        buffer1 = self.manager.get_buffer(self.identity)
        buffer2 = self.manager.get_buffer(self.identity)
        assert buffer1 is buffer2

    def test_add_message(self):
        """测试添加消息"""
        msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        
        self.manager.add_message(self.identity, msg)
        
        buffer = self.manager.get_buffer(self.identity)
        assert buffer.message_count == 1
        assert buffer.messages[0] is msg

    def test_clear_buffer(self):
        """测试清空 buffer"""
        msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        self.manager.add_message(self.identity, msg)
        
        result = self.manager.clear_buffer(self.identity)
        
        assert result is True
        buffer = self.manager.get_buffer(self.identity)
        assert buffer.message_count == 0

    def test_clear_nonexistent_buffer(self):
        """测试清空不存在的 buffer"""
        result = self.manager.clear_buffer(self.identity)
        assert result is False

    def test_list_active_buffers(self):
        """测试列出活跃 buffers"""
        self.manager.get_buffer(self.identity)
        
        active = self.manager.list_active_buffers()
        assert len(active) == 1
        assert self.identity.buffer_key in active

    def test_get_buffer_info(self):
        """测试获取 buffer 信息"""
        msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="Hello"
        )
        self.manager.add_message(self.identity, msg)
        
        info = self.manager.get_buffer_info(self.identity)
        buffer = self.manager.get_buffer(self.identity)
        
        assert info["exists"] is True
        assert info["buffer_id"] == buffer.buffer_id
        assert info["message_count"] == 1
        assert info["user_id"] == "user1"
        assert info["agent_id"] == "agent1"
        assert info["session_id"] == "session1"

    def test_get_buffer_info_nonexistent(self):
        """测试获取不存在的 buffer 信息"""
        info = self.manager.get_buffer_info(self.identity)
        assert info["exists"] is False
