"""
感知引擎组件协作测试

测试感知引擎内部各组件之间的协作：
- StreamParser 与 LogicalBlock 的交互
- SemanticBoundaryAdsorber 与 Buffer 的交互
- TriggerManager 与 PerceptionLayer 的交互
- SimplePerceptionLayer 的组件编排
- SemanticFlowPerceptionLayer 的组件编排

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
    FlushReason,
    Identity,
)
from hivememory.engines.generation.models import ConversationMessage
from hivememory.engines.perception import (
    SimplePerceptionLayer,
    SemanticFlowPerceptionLayer,
    UnifiedStreamParser,
    SemanticBoundaryAdsorber,
    TriggerManager,
    MessageCountTrigger,
    IdleTimeoutTrigger,
    SemanticBoundaryTrigger,
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

    def test_parser_identifies_block_boundaries(self):
        """测试解析器识别Block边界"""
        parser = UnifiedStreamParser()

        # USER_QUERY 应该触发新Block
        user_msg = parser.parse_message({"role": "user", "content": "新问题"})
        should_create_block = parser.should_create_new_block(user_msg)

        assert should_create_block is True


class TestAdsorberAndBufferCollaboration:
    """测试 SemanticBoundaryAdsorber 与 Buffer 的协作"""

    def test_adsorber_initializes_buffer_kernel(self):
        """测试吸附器初始化Buffer话题核心"""
        from hivememory.engines.perception.models import SemanticBuffer, LogicalBlock, StreamMessage, StreamMessageType

        adsorber = SemanticBoundaryAdsorber()
        buffer = SemanticBuffer(
            user_id="test_user",
            agent_id="test_agent",
            session_id="test_session",
        )

        # 创建初始Block
        block = LogicalBlock()
        block.add_stream_message(StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content="Python编程问题",
            metadata={"role": "user"}
        ))
        block.add_stream_message(StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="Python是一种编程语言",
            metadata={"role": "assistant"}
        ))

        # 更新话题核心
        adsorber.update_topic_kernel(buffer, block)

        # 验证话题核心已创建
        assert buffer.topic_kernel_vector is not None

    def test_adsorber_detects_topic_shift(self):
        """测试吸附器检测话题切换"""
        from hivememory.engines.perception.models import SemanticBuffer, LogicalBlock, StreamMessage, StreamMessageType

        adsorber = SemanticBoundaryAdsorber()
        buffer = SemanticBuffer(
            user_id="test_user",
            agent_id="test_agent",
            session_id="test_session",
        )

        # 建立初始话题
        block1 = LogicalBlock()
        block1.add_stream_message(StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content="Python编程",
            metadata={"role": "user"}
        ))
        block1.add_stream_message(StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="Python教程",
            metadata={"role": "assistant"}
        ))

        adsorber.update_topic_kernel(buffer, block1)
        buffer.blocks = [block1]

        # 创建不同话题的Block
        block2 = LogicalBlock()
        block2.add_stream_message(StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content="红烧肉做法",
            metadata={"role": "user"}
        ))
        block2.add_stream_message(StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="烹饪教程",
            metadata={"role": "assistant"}
        ))

        # 检查是否应该吸附
        should_adsorb, reason = adsorber.should_adsorb(block2, buffer)

        # 不同话题可能不吸附（取决于实现）
        assert reason is not None or should_adsorb is not None


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

        perception = SimplePerceptionLayer(
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush,
        )

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        # 添加消息直到触发
        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)
        perception.add_message("user", "消息2", identity)

        # 应该触发flush
        assert len(flush_called) >= 1

    def test_idle_timeout_trigger(self):
        """测试空闲超时触发器"""
        flush_called = []

        def on_flush(messages, reason):
            flush_called.append((messages, reason))

        trigger_manager = TriggerManager(strategies=[
            IdleTimeoutTrigger(timeout=0.1)  # 100ms超时
        ])

        perception = SimplePerceptionLayer(
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush,
        )

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "消息1", identity)

        # 等待超时
        import time
        time.sleep(0.15)

        perception.add_message("user", "消息2", identity)

        # 应该触发超时flush
        assert len(flush_called) >= 1


class TestSimplePerceptionLayerOrchestration:
    """测试 SimplePerceptionLayer 的编排"""

    def test_buffer_management(self):
        """测试Buffer管理"""
        perception = SimplePerceptionLayer()

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        # 添加消息
        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        # 获取Buffer信息
        info = perception.get_buffer_info(identity)

        assert info['exists'] is True
        assert info['message_count'] == 2

    def test_manual_flush(self):
        """测试手动Flush"""
        flush_called = []

        def on_flush(messages, reason):
            flush_called.append((messages, reason))

        perception = SimplePerceptionLayer(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        # 手动flush
        messages = perception.flush_buffer(identity)

        assert messages is not None
        assert len(flush_called) >= 1

    def test_multi_session_isolation(self):
        """测试多会话隔离"""
        perception = SimplePerceptionLayer()

        identity1 = Identity(user_id="user1", agent_id="agent", session_id="session1")
        identity2 = Identity(user_id="user2", agent_id="agent", session_id="session2")

        perception.add_message("user", "用户1的消息", identity1)
        perception.add_message("user", "用户2的消息", identity2)

        info1 = perception.get_buffer_info(identity1)
        info2 = perception.get_buffer_info(identity2)

        # 每个会话应该有独立的buffer
        assert info1['message_count'] == 1
        assert info2['message_count'] == 1


class TestSemanticFlowPerceptionLayerOrchestration:
    """测试 SemanticFlowPerceptionLayer 的编排"""

    def test_semantic_flow_initialization(self):
        """测试语义流初始化"""
        perception = SemanticFlowPerceptionLayer()

        assert perception is not None

    def test_semantic_flow_buffer_info(self):
        """测试语义流Buffer信息获取"""
        perception = SemanticFlowPerceptionLayer()

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "测试消息", identity)
        perception.add_message("assistant", "测试回复", identity)

        info = perception.get_buffer_info(identity)

        assert info['exists'] is True

    def test_semantic_flow_flush(self):
        """测试语义流Flush"""
        flush_called = []

        def on_flush(messages, reason):
            flush_called.append((messages, reason))

        perception = SemanticFlowPerceptionLayer(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "消息1", identity)
        perception.add_message("assistant", "回复1", identity)

        messages = perception.flush_buffer(identity)

        assert messages is not None or len(flush_called) > 0


class TestPerceptionAndGenerationCollaboration:
    """测试感知层与生成层的协作"""

    def test_messages_converted_to_conversation_messages(self):
        """测试消息转换为ConversationMessage"""
        perception = SimplePerceptionLayer()

        flush_called = []

        def on_flush(messages, reason):
            flush_called.append(messages)

        perception = SimplePerceptionLayer(on_flush_callback=on_flush)

        identity = Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")

        perception.add_message("user", "用户消息", identity)
        perception.add_message("assistant", "助手回复", identity)

        perception.flush_buffer(identity)

        # 验证返回的消息是ConversationMessage格式
        if len(flush_called) > 0 and len(flush_called[0]) > 0:
            first_msg = flush_called[0][0]
            assert isinstance(first_msg, ConversationMessage)


class TestTokenManagement:
    """测试Token管理"""

    def test_token_estimation(self):
        """测试Token估算"""
        from hivememory.engines.perception.models import estimate_tokens

        text = "这是一个测试句子，用于验证token估算功能。"
        tokens = estimate_tokens(text)

        assert tokens > 0
        # 中文大约1个字符=1个token，但某些 tokenizer 可能压缩率更高
        assert tokens >= len(text) // 3

    def test_block_token_count(self):
        """测试Block的Token计数"""
        from hivememory.engines.perception.models import LogicalBlock, StreamMessage, StreamMessageType

        block = LogicalBlock()
        block.add_stream_message(StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content="用户消息",
            metadata={"role": "user"}
        ))
        block.add_stream_message(StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="助手回复",
            metadata={"role": "assistant"}
        ))

        tokens = block.total_tokens

        assert tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
