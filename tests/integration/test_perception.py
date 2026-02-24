"""
感知引擎组件协作测试

测试感知引擎内部各组件之间的协作：
- SemanticBoundaryAdsorber 与 Buffer 的协作
- TriggerManager 与 PerceptionLayer 的交互
- SimplePerceptionLayer 的组件编排
- SemanticFlowPerceptionLayer 的组件编排

Note:
    v3.0 重构：
    - perceive() 已移除，统一使用 ingest_payload()
    - UnifiedStreamParser 已移除，使用 MTPLogParser 替代
    - Adsorber.should_adsorb() 返回 Optional[FlushEvent]
    - Adsorber.compute_new_topic_kernel() 替代 update_topic_kernel()

不测试：与外部服务（LLM、Embedding）的交互
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from hivememory.core.models import (
    Identity,
    StreamMessageType,
    StreamMessage,
)
from hivememory.engines.perception import (
    SimplePerceptionLayer,
    SemanticFlowPerceptionLayer,
    SemanticBoundaryAdsorber,
    TriggerManager,
    MessageCountTrigger,
    SemanticBoundaryTrigger,
    FlushEvent,
)
from hivememory.engines.perception.models import (
    LogicalBlock,
    SemanticBuffer,
    FlushReason,
    InteractionPayload,
)
from hivememory.patchouli.config import (
    SimplePerceptionConfig,
    SemanticFlowPerceptionConfig,
    SemanticAdsorberConfig,
)


def _make_payload(user_msg, assistant_msg, identity):
    """辅助: 构建 InteractionPayload"""
    return InteractionPayload(
        user_message=user_msg,
        assistant_message=assistant_msg,
        identity=identity,
    )


class TestAdsorberAndBufferCollaboration:
    """测试 SemanticBoundaryAdsorber 与 Buffer 的协作"""

    def test_adsorber_computes_buffer_kernel(self):
        """测试吸附器计算 Buffer 话题核心"""
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [0.1, 0.2, 0.3]

        config = SemanticAdsorberConfig()
        adsorber = SemanticBoundaryAdsorber(config=config, embedding_service=mock_embedding)
        buffer = SemanticBuffer(
            identity=Identity(user_id="test_user", agent_id="test_agent", session_id="test_session"),
        )

        block = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="Python编程问题",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="Python是一种编程语言",
            )
        )

        new_kernel = adsorber.compute_new_topic_kernel(buffer, block)

        assert new_kernel is not None
        assert len(new_kernel) == 3

    def test_adsorber_detects_topic_shift(self):
        """测试吸附器检测话题切换"""
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [0.1, 0.2, 0.3]
        mock_embedding.compute_cosine_similarity.return_value = 0.1

        config = SemanticAdsorberConfig()
        adsorber = SemanticBoundaryAdsorber(config=config, embedding_service=mock_embedding)
        buffer = SemanticBuffer(
            identity=Identity(user_id="test_user", agent_id="test_agent", session_id="test_session"),
        )

        block1 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER, content="Python编程",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT, content="Python教程",
            )
        )

        buffer.blocks.append(block1)
        buffer.topic_kernel_vector = [0.9, 0.1, 0.0]

        block2 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER, content="红烧肉做法",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT, content="烹饪教程",
            ),
            rewritten_query="红烧肉做法",
        )

        result = adsorber.should_adsorb(buffer, block2)

        assert result is not None
        assert isinstance(result, FlushEvent)
        assert result.flush_reason == FlushReason.SEMANTIC_DRIFT


class TestTriggerAndPerceptionCollaboration:
    """测试 TriggerManager 与 PerceptionLayer 的协作"""

    def test_message_count_trigger(self):
        """测试消息数触发器"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append((messages, reason))

        trigger_manager = TriggerManager(strategies=[
            MessageCountTrigger(threshold=3)
        ])

        config = SimplePerceptionConfig()
        perception = SimplePerceptionLayer(
            config=config,
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush,
        )

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        # ingest_payload 添加 2 条消息 (user + assistant)
        perception.ingest_payload(_make_payload("消息1", "回复1", identity))
        # 再添加 2 条，总共 4 条，超过阈值 3
        perception.ingest_payload(_make_payload("消息2", "回复2", identity))

        assert len(flush_called) >= 1


class TestSimplePerceptionLayerOrchestration:
    """测试 SimplePerceptionLayer 的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建 SimplePerceptionLayer"""
        config = SimplePerceptionConfig()
        trigger_manager = TriggerManager(strategies=[
            MessageCountTrigger(threshold=10)
        ])
        return SimplePerceptionLayer(
            config=config,
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush_callback,
        )

    def test_buffer_management(self):
        """测试 Buffer 管理"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.ingest_payload(_make_payload("消息1", "回复1", identity))

        info = perception.get_buffer_info(identity)

        assert info['exists'] is True
        assert info['message_count'] == 2

    def test_manual_flush(self):
        """测试手动 Flush"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append((messages, reason))

        perception = self._create_perception(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.ingest_payload(_make_payload("消息1", "回复1", identity))

        messages = perception.flush_buffer(identity)

        assert messages is not None
        assert len(flush_called) >= 1

    def test_multi_session_isolation(self):
        """测试多会话隔离"""
        perception = self._create_perception()

        identity1 = Identity(user_id="user1", agent_id="agent", session_id="session1")
        identity2 = Identity(user_id="user2", agent_id="agent", session_id="session2")

        perception.ingest_payload(_make_payload("用户1的消息", "回复1", identity1))
        perception.ingest_payload(_make_payload("用户2的消息", "回复2", identity2))

        info1 = perception.get_buffer_info(identity1)
        info2 = perception.get_buffer_info(identity2)

        assert info1['message_count'] == 2
        assert info2['message_count'] == 2


class TestSemanticFlowPerceptionLayerOrchestration:
    """测试 SemanticFlowPerceptionLayer 的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建 SemanticFlowPerceptionLayer"""
        config = SemanticFlowPerceptionConfig()

        mock_adsorber = Mock()
        mock_adsorber.should_adsorb.return_value = None
        mock_adsorber.compute_new_topic_kernel.return_value = [0.1, 0.2, 0.3]

        mock_relay = Mock()
        mock_relay.should_relay.return_value = None

        return SemanticFlowPerceptionLayer(
            config=config,
            adsorber=mock_adsorber,
            relay_controller=mock_relay,
            on_flush_callback=on_flush_callback,
        )

    def test_semantic_flow_initialization(self):
        """测试语义流初始化"""
        perception = self._create_perception()
        assert perception is not None

    def test_semantic_flow_buffer_info(self):
        """测试语义流 Buffer 信息获取"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.ingest_payload(_make_payload("测试消息", "测试回复", identity))

        info = perception.get_buffer_info(identity)

        assert info['exists'] is True

    def test_semantic_flow_flush(self):
        """测试语义流 Flush"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append((messages, reason))

        perception = self._create_perception(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.ingest_payload(_make_payload("消息1", "回复1", identity))

        messages = perception.flush_buffer(identity)

        assert messages is not None or len(flush_called) > 0


class TestPerceptionAndGenerationCollaboration:
    """测试感知层与生成层的协作"""

    def test_messages_converted_to_stream_messages(self):
        """测试消息转换为 StreamMessage"""
        flush_called = []

        def on_flush(messages, reason, **kwargs):
            flush_called.append(messages)

        config = SimplePerceptionConfig()
        trigger_manager = TriggerManager(strategies=[
            MessageCountTrigger(threshold=10)
        ])
        perception = SimplePerceptionLayer(
            config=config,
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush,
        )

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.ingest_payload(_make_payload("用户消息", "助手回复", identity))

        perception.flush_buffer(identity)

        if len(flush_called) > 0 and len(flush_called[0]) > 0:
            first_msg = flush_called[0][0]
            assert isinstance(first_msg, StreamMessage)
            assert first_msg.message_type == StreamMessageType.USER


class TestTokenManagement:
    """测试 Token 管理"""

    def test_token_estimation(self):
        """测试 Token 估算"""
        from hivememory.engines.perception.models import estimate_tokens

        text = "这是一个测试句子，用于验证token估算功能。"
        tokens = estimate_tokens(text)

        assert tokens > 0
        assert tokens >= len(text) // 3

    def test_block_token_count(self):
        """测试 Block 的 Token 计数"""
        block = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="用户消息",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="助手回复",
            ),
            total_tokens=100,
        )

        tokens = block.total_tokens

        assert tokens == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
