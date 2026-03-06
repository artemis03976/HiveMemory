"""
BufferManager 单元测试

测试覆盖:
- SemanticBufferManager (原 BufferManager)
    - Buffer 获取和创建 (create_buffer, get_buffer)
    - Buffer CRUD 操作 (pop_buffer)
    - Buffer 元数据更新
    - 多会话/多话题隔离

Note:
    Phase 4.5 重构：BufferManager 从 Identity-based 改为 topic_id-based
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from hivememory.core.models import Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
)


class TestSemanticBufferManagerBasic:
    """SemanticBufferManager 基础功能测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
        )

    # ========== Buffer 获取和创建测试 ==========

    def test_get_buffer_returns_none_if_not_exists(self):
        """测试获取不存在的 buffer 返回 None"""
        buffer = self.manager.get_buffer("nonexistent_topic")
        assert buffer is None

    def test_create_buffer(self):
        """测试创建新 buffer"""
        buffer = self.manager.create_buffer(self.identity, title="Test Topic")

        assert buffer is not None
        assert buffer.identity.user_id == "user1"
        assert buffer.identity.agent_id == "agent1"
        assert buffer.title == "Test Topic"
        assert buffer.topic_id is not None
        assert len(buffer.blocks) == 0

    def test_get_buffer_returns_existing(self):
        """测试获取已存在的 buffer"""
        created = self.manager.create_buffer(self.identity)
        buffer = self.manager.get_buffer(created.topic_id)

        assert buffer is created
        assert buffer.identity == self.identity


class TestSemanticBufferManagerCRUD:
    """SemanticBufferManager CRUD 操作测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(
            user_id="user1",
            agent_id="agent1",
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

    # ========== add_block 测试 ==========

    def test_add_block(self):
        """测试添加 block 到 buffer"""
        buffer = self.manager.create_buffer(self.identity)
        block = self._create_block()

        self.manager.add_block(buffer.topic_id, block)

        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert retrieved is not None
        assert len(retrieved.blocks) == 1
        assert retrieved.blocks[0] is block
        assert retrieved.total_tokens == 100

    def test_add_multiple_blocks(self):
        """测试添加多个 blocks"""
        buffer = self.manager.create_buffer(self.identity)
        block1 = self._create_block("Hello 1")
        block2 = self._create_block("Hello 2")

        self.manager.add_block(buffer.topic_id, block1)
        self.manager.add_block(buffer.topic_id, block2)

        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert len(retrieved.blocks) == 2
        assert retrieved.total_tokens == 200

    # ========== pop_buffer 测试 ==========

    def test_pop_buffer_returns_buffer(self):
        """测试移除 buffer 并返回"""
        buffer = self.manager.create_buffer(self.identity)
        block = self._create_block("Hello 1")
        self.manager.add_block(buffer.topic_id, block)

        popped = self.manager.pop_buffer(buffer.topic_id)

        assert popped is not None
        assert len(popped.blocks) == 1
        assert popped.blocks[0] is block

        # 确认已从 manager 中移除
        assert self.manager.get_buffer(buffer.topic_id) is None

    def test_pop_nonexistent_buffer_returns_none(self):
        """测试移除不存在的 buffer 返回 None"""
        popped = self.manager.pop_buffer("nonexistent_topic")
        assert popped is None

    # ========== clear_buffer 测试 ==========

    def test_clear_buffer_returns_blocks(self):
        """测试清空 buffer 内容并保留 buffer"""
        buffer = self.manager.create_buffer(self.identity)
        block = self._create_block("Hello")
        self.manager.add_block(buffer.topic_id, block)

        cleared = self.manager.clear_buffer(buffer.topic_id)

        assert len(cleared) == 1
        assert cleared[0] is block

        # 确认 buffer 仍然存在但为空
        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert retrieved is not None
        assert len(retrieved.blocks) == 0
        assert retrieved.total_tokens == 0

    def test_clear_nonexistent_buffer_returns_empty(self):
        """测试清空不存在的 buffer 返回空列表"""
        cleared = self.manager.clear_buffer("nonexistent_topic")
        assert cleared == []

    # ========== update_metadata 测试 ==========

    def test_update_topic_kernel_vector(self):
        """测试更新话题核心向量"""
        buffer = self.manager.create_buffer(self.identity)

        new_vector = [0.1, 0.2, 0.3]
        self.manager.update_metadata(
            buffer.topic_id,
            topic_kernel_vector=new_vector
        )

        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert retrieved.topic_kernel_vector == new_vector

    def test_update_state(self):
        """测试更新状态"""
        buffer = self.manager.create_buffer(self.identity)

        self.manager.update_metadata(
            buffer.topic_id,
            state=BufferState.PROCESSING
        )

        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert retrieved.state == BufferState.PROCESSING

    def test_update_multiple_fields(self):
        """测试同时更新多个字段"""
        buffer = self.manager.create_buffer(self.identity)

        self.manager.update_metadata(
            buffer.topic_id,
            topic_kernel_vector=[0.1, 0.2],
            state=BufferState.FLUSHING
        )

        retrieved = self.manager.get_buffer(buffer.topic_id)
        assert retrieved.topic_kernel_vector == [0.1, 0.2]
        assert retrieved.state == BufferState.FLUSHING


class TestSemanticBufferManagerMultiTopic:
    """SemanticBufferManager 多话题测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()

    def test_multi_topic_isolation(self):
        """测试多话题隔离"""
        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建两个话题
        buffer1 = self.manager.create_buffer(identity, title="Topic 1")
        buffer2 = self.manager.create_buffer(identity, title="Topic 2")

        # 两个 buffer 应该有不同的 topic_id
        assert buffer1.topic_id != buffer2.topic_id
        assert buffer1.title == "Topic 1"
        assert buffer2.title == "Topic 2"

        # 可以通过 topic_id 独立获取
        retrieved1 = self.manager.get_buffer(buffer1.topic_id)
        retrieved2 = self.manager.get_buffer(buffer2.topic_id)

        assert retrieved1 is buffer1
        assert retrieved2 is buffer2

    def test_get_active_topic_buffer_count(self):
        """测试获取活跃 buffer 数量"""
        identity1 = Identity(user_id="user1", agent_id="agent1")
        identity2 = Identity(user_id="user2", agent_id="agent1")

        self.manager.create_buffer(identity1)
        self.manager.create_buffer(identity2)

        count = self.manager.get_active_topic_buffer_count()
        assert count == 2

    def test_get_all_buffers(self):
        """测试获取所有活跃 buffer"""
        identity = Identity(user_id="user1", agent_id="agent1")
        buffer1 = self.manager.create_buffer(identity)
        buffer2 = self.manager.create_buffer(identity)

        buffers = self.manager.get_all_buffers()
        assert len(buffers) == 2
        topic_ids = [b.topic_id for b in buffers]
        assert buffer1.topic_id in topic_ids
        assert buffer2.topic_id in topic_ids


class TestSemanticBufferManagerInfo:
    """SemanticBufferManager 信息查询测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()
        self.identity = Identity(user_id="user1", agent_id="agent1")

    def test_get_buffer_info_nonexistent(self):
        """测试获取不存在 buffer 的信息"""
        info = self.manager.get_buffer_info("nonexistent_topic")
        assert info["exists"] is False

    def test_get_buffer_info_existing(self):
        """测试获取存在 buffer 的信息"""
        buffer = self.manager.create_buffer(self.identity)

        info = self.manager.get_buffer_info(buffer.topic_id)

        assert info["exists"] is True
        assert info["block_count"] == 0
        assert info["total_tokens"] == 0
        assert info["state"] == "idle"

    def test_get_buffer_info_with_blocks(self):
        """测试获取有 blocks 的 buffer 信息"""
        buffer = self.manager.create_buffer(self.identity)

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

        self.manager.add_block(buffer.topic_id, block)

        info = self.manager.get_buffer_info(buffer.topic_id)

        assert info["exists"] is True
        assert info["block_count"] == 1
        assert info["total_tokens"] == 50

    def test_get_buffer_info_with_topic_kernel(self):
        """测试获取有话题核心的 buffer 信息"""
        buffer = self.manager.create_buffer(self.identity)
        self.manager.update_metadata(
            buffer.topic_id,
            topic_kernel_vector=[0.1, 0.2, 0.3]
        )

        info = self.manager.get_buffer_info(buffer.topic_id)

        assert info["has_topic_kernel"] is True


class TestSemanticBufferMenu:
    """活跃话题菜单测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.manager = SemanticBufferManager()

    def test_get_active_topics_menu(self):
        """测试获取活跃话题菜单"""
        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建话题
        buf1 = self.manager.create_buffer(identity, title="Topic 1")
        buf2 = self.manager.create_buffer(identity, title="Topic 2")

        # 添加 blocks 以使话题出现在菜单中
        block = LogicalBlock(
            user_query="test",
            clean_response="test",
            total_tokens=10
        )
        self.manager.add_block(buf1.topic_id, block)

        menu = self.manager.get_active_topics_menu()

        # 只有有内容的话题出现在菜单中
        assert len(menu) == 1
        assert menu[0]["topic_id"] == buf1.topic_id
        assert menu[0]["title"] == "Topic 1"
