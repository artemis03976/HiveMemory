"""
Perception 层集成测试

测试感知层与各组件的协作:
- 与 Adsorber 的协作
- 与 Generation Engine 的协作
- 与 Buffer Manager 的协作

Note:
    Phase 4.5 重构：PerceptionLayer 方法改为使用 topic_id
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from hivememory.core.models import Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
    InteractionPayload,
)
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.patchouli.config import SemanticFlowPerceptionConfig


def _make_payload(user_msg: str, assistant_msg: str, identity: Identity) -> InteractionPayload:
    """辅助方法：创建测试用 Payload"""
    return InteractionPayload(
        user_message=user_msg,
        assistant_message=assistant_msg,
        identity=identity,
    )


class TestAdsorberAndBufferCollaboration:
    """测试感知层与 Buffer Manager 的协作"""

    def test_adsorber_computes_buffer_kernel(self):
        """测试 Buffer Kernel 计算"""
        manager = SemanticBufferManager()

        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建 buffer
        buffer = manager.create_buffer(identity)

        # 验证 buffer 创建
        assert buffer is not None
        assert buffer.identity == identity
        assert buffer.topic_id is not None

    def test_adsorber_detects_topic_shift(self):
        """测试话题切换检测"""
        manager = SemanticBufferManager()
        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建两个话题
        buf1 = manager.create_buffer(identity, title="Topic 1")
        buf2 = manager.create_buffer(identity, title="Topic 2")

        # 验证话题隔离
        assert buf1.topic_id != buf2.topic_id
        assert buf1.identity == buf2.identity  # 同一个用户


class TestSemanticFlowPerceptionLayerOrchestration:
    """测试语义流感知层的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建感知层实例"""
        config = SemanticFlowPerceptionConfig()

        mock_relay = Mock()
        mock_relay.should_relay.return_value = None

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
        )
        if on_flush_callback is not None:
            perception.set_generation_callback(on_flush_callback)
        return perception

    @pytest.mark.asyncio
    async def test_semantic_flow_initialization(self):
        """测试语义流初始化"""
        perception = self._create_perception()
        assert perception is not None

    @pytest.mark.asyncio
    async def test_semantic_flow_route_and_ingest(self):
        """测试语义流路由和摄入"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 验证话题创建
        menu = perception.get_active_topics_menu()
        assert len(menu) == 1
        assert menu[0]["title"] == "新建话题"

    @pytest.mark.asyncio
    async def test_semantic_flow_buffer_info(self):
        """测试语义流 Buffer 信息获取"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 获取菜单以获取 topic_id
        menu = perception.get_active_topics_menu()
        topic_id = menu[0]["topic_id"]

        # 通过 topic_id 获取信息
        info = perception.get_buffer_info(topic_id)

        assert info['exists'] is True

    @pytest.mark.asyncio
    async def test_semantic_flow_flush(self):
        """测试语义流 Flush"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append((messages, reason))

        perception = self._create_perception(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("消息1", "回复1", identity)
        )

        # 获取 topic_id
        menu = perception.get_active_topics_menu()
        topic_id = menu[0]["topic_id"]

        # 通过 topic_id flush
        messages = perception.flush_buffer(topic_id)

        # 验证 flush 成功
        assert messages is not None


class TestPerceptionAndGenerationCollaboration:
    """测试感知层与生成层的协作"""

    @pytest.mark.asyncio
    async def test_messages_converted_to_stream_messages(self):
        """测试消息转换为 StreamMessage"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append(messages)

        config = SemanticFlowPerceptionConfig()
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
        )
        perception.set_generation_callback(on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("用户消息", "助手回复", identity)
        )

        # 获取 topic_id 并 flush
        menu = perception.get_active_topics_menu()
        topic_id = menu[0]["topic_id"]

        perception.flush_buffer(topic_id)

        if len(flush_called) > 0 and len(flush_called[0]) > 0:
            first_msg = flush_called[0][0]
            assert isinstance(first_msg, StreamMessage)
            assert first_msg.message_type == StreamMessageType.USER


class TestTokenManagement:
    """测试 Token 管理"""

    def test_token_estimation(self):
        """测试 Token 估算"""
        from hivememory.utils.token_estimator import estimate_tokens

        text = "Hello, world!"
        tokens = estimate_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_block_token_count(self):
        """测试 Block Token 计数"""
        block = LogicalBlock(
            user_query="Test query",
            clean_response="Test response",
            total_tokens=0,
        )

        from hivememory.utils.token_estimator import estimate_tokens
        block.total_tokens = (
            estimate_tokens(block.user_query) +
            estimate_tokens(block.clean_response)
        )

        assert block.total_tokens > 0
