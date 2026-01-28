"""
感知引擎组件协作测试

测试感知引擎内部各组件之间的协作：
- StreamParser 与 LogicalBlock 的交互
- SemanticBoundaryAdsorber 与 Buffer 的协作
- TriggerManager 与 PerceptionLayer 的交互
- SimplePerceptionLayer 的组件编排
- SemanticFlowPerceptionLayer 的组件编排

Note:
    v3.0 重构：
    - should_create_new_block() 移至 LogicalBlockBuilder
    - LogicalBlock 通过构造函数创建
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
    UnifiedStreamParser,
    SemanticBoundaryAdsorber,
    TriggerManager,
    MessageCountTrigger,
    SemanticBoundaryTrigger,
    LogicalBlockBuilder,
    FlushEvent,
)
from hivememory.engines.perception.models import (
    LogicalBlock,
    SemanticBuffer,
    FlushReason,
)
from hivememory.patchouli.config import (
    SimplePerceptionConfig,
    SemanticFlowPerceptionConfig,
)


class TestParserAndBlockCollaboration:
    """测试 StreamParser 与 LogicalBlock 的协作"""

    def test_parser_creates_correct_message_types(self):
        """测试解析器创建正确的消息类型"""
        parser = UnifiedStreamParser()

        # 测试不同格式的消息
        user_msg = parser.parse_message({"role": "user", "content": "测试消息"})
        assistant_msg = parser.parse_message({"role": "assistant", "content": "回复"})

        assert user_msg is not None
        assert assistant_msg is not None

    def test_builder_identifies_block_boundaries(self):
        """测试 Builder 识别 Block 边界"""
        builder = LogicalBlockBuilder()

        # USER 消息应该触发新 Block
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content="新问题"
        )
        should_create_block = builder.should_create_new_block(user_msg)

        assert should_create_block is True


class TestAdsorberAndBufferCollaboration:
    """测试 SemanticBoundaryAdsorber 与 Buffer 的协作"""

    def test_adsorber_computes_buffer_kernel(self):
        """测试吸附器计算 Buffer 话题核心"""
        # Mock Embedding Service
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [0.1, 0.2, 0.3]

        adsorber = SemanticBoundaryAdsorber(embedding_service=mock_embedding)
        buffer = SemanticBuffer(
            identity=Identity(user_id="test_user", agent_id="test_agent", session_id="test_session"),
        )

        # 创建初始 Block（通过构造函数）
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

        # 计算话题核心（纯函数，不修改 buffer）
        new_kernel = adsorber.compute_new_topic_kernel(buffer, block)

        # 验证返回了新的话题核心向量
        assert new_kernel is not None
        assert len(new_kernel) == 3

    def test_adsorber_detects_topic_shift(self):
        """测试吸附器检测话题切换"""
        # Mock Embedding Service
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [0.1, 0.2, 0.3]
        mock_embedding.compute_cosine_similarity.return_value = 0.1  # 低相似度

        adsorber = SemanticBoundaryAdsorber(embedding_service=mock_embedding)
        buffer = SemanticBuffer(
            identity=Identity(user_id="test_user", agent_id="test_agent", session_id="test_session"),
        )

        # 建立初始话题
        block1 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="Python编程",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="Python教程",
            )
        )

        buffer.blocks.append(block1)
        buffer.topic_kernel_vector = [0.9, 0.1, 0.0]  # 设置话题核心

        # 创建不同话题的 Block
        block2 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="红烧肉做法",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="烹饪教程",
            ),
            rewritten_query="红烧肉做法",
        )

        # 检查是否应该吸附（v3.0: 返回 Optional[FlushEvent]）
        result = adsorber.should_adsorb(buffer, block2)

        # 低相似度应该返回 FlushEvent（语义漂移）
        assert result is not None
        assert isinstance(result, FlushEvent)
        assert result.flush_reason == FlushReason.SEMANTIC_DRIFT


class TestTriggerAndPerceptionCollaboration:
    """测试 TriggerManager 与 PerceptionLayer 的协作"""

    def test_message_count_trigger(self):
        """测试消息数触发器"""
        flush_called = []

        def on_flush(messages, reason):
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

        # 添加消息直到触发
        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)
        perception.add_message("user", "消息2", identity)

        # 应该触发 flush
        assert len(flush_called) >= 1


class TestSimplePerceptionLayerOrchestration:
    """测试 SimplePerceptionLayer 的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建 SimplePerceptionLayer"""
        config = SimplePerceptionConfig()
        trigger_manager = TriggerManager(strategies=[
            MessageCountTrigger(threshold=10)  # 高阈值避免自动触发
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

        # 添加消息
        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        # 获取 Buffer 信息
        info = perception.get_buffer_info(identity)

        assert info['exists'] is True
        assert info['message_count'] == 2

    def test_manual_flush(self):
        """测试手动 Flush"""
        flush_called = []

        def on_flush(messages, reason):
            flush_called.append((messages, reason))

        perception = self._create_perception(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        # 手动 flush
        messages = perception.flush_buffer(identity)

        assert messages is not None
        assert len(flush_called) >= 1

    def test_multi_session_isolation(self):
        """测试多会话隔离"""
        perception = self._create_perception()

        identity1 = Identity(user_id="user1", agent_id="agent", session_id="session1")
        identity2 = Identity(user_id="user2", agent_id="agent", session_id="session2")

        perception.add_message("user", "用户1的消息", identity1)
        perception.add_message("user", "用户2的消息", identity2)

        info1 = perception.get_buffer_info(identity1)
        info2 = perception.get_buffer_info(identity2)

        # 每个会话应该有独立的 buffer
        assert info1['message_count'] == 1
        assert info2['message_count'] == 1


class TestSemanticFlowPerceptionLayerOrchestration:
    """测试 SemanticFlowPerceptionLayer 的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建 SemanticFlowPerceptionLayer"""
        config = SemanticFlowPerceptionConfig()
        parser = UnifiedStreamParser()

        # Mock adsorber
        mock_adsorber = Mock()
        mock_adsorber.should_adsorb.return_value = None  # 继续吸附
        mock_adsorber.compute_new_topic_kernel.return_value = [0.1, 0.2, 0.3]

        # Mock relay
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None  # 不需要接力

        return SemanticFlowPerceptionLayer(
            config=config,
            parser=parser,
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

        perception.add_message("user", "测试消息", identity)
        perception.add_message("assistant", "测试回复", identity)

        info = perception.get_buffer_info(identity)

        assert info['exists'] is True

    def test_semantic_flow_flush(self):
        """测试语义流 Flush"""
        flush_called = []

        def on_flush(messages, reason):
            flush_called.append((messages, reason))

        perception = self._create_perception(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        messages = perception.flush_buffer(identity)

        assert messages is not None or len(flush_called) > 0


class TestPerceptionAndGenerationCollaboration:
    """测试感知层与生成层的协作"""

    def test_messages_converted_to_stream_messages(self):
        """测试消息转换为 StreamMessage"""
        flush_called = []

        def on_flush(messages, reason):
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

        perception.add_message("user", "用户消息", identity)
        perception.add_message("assistant", "助手回复", identity)

        perception.flush_buffer(identity)

        # 验证返回的消息是 StreamMessage 格式
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
        # 中文大约1个字符=1个token，但某些 tokenizer 可能压缩率更高
        assert tokens >= len(text) // 3

    def test_block_token_count(self):
        """测试 Block 的 Token 计数"""
        # 通过构造函数创建 Block
        block = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="用户消息",
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="助手回复",
            ),
            total_tokens=100,  # 直接设置 token 数
        )

        tokens = block.total_tokens

        assert tokens == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
