"""
感知层 (Perception Layers) 单元测试

测试覆盖:
- SimplePerceptionLayer:
    - 消息添加流程
    - 触发器调用
    - Flush 回调
- SemanticFlowPerceptionLayer:
    - 流式消息解析流程
    - Block 管理
    - 语义吸附流程
    - Flush 回调

Note:
    v3.0 重构：
    - SemanticFlowPerceptionLayer 现在负责编排 flush 逻辑
    - Adsorber.should_adsorb() 返回 Optional[FlushEvent]
    - Relay.should_relay() 返回 Optional[FlushEvent]
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.perception.simple_perception_layer import SimplePerceptionLayer
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.models import (
    FlushEvent,
    SimpleBuffer,
    SemanticBuffer,
    LogicalBlock,
    FlushReason,
)
from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.patchouli.config import SimplePerceptionConfig, SemanticFlowPerceptionConfig


class TestSimplePerceptionLayer:
    """测试简单感知层"""

    def setup_method(self):
        self.mock_trigger_manager = Mock()
        self.mock_callback = Mock()
        self.config = SimplePerceptionConfig()
        self.layer = SimplePerceptionLayer(
            config=self.config,
            trigger_manager=self.mock_trigger_manager,
            on_flush_callback=self.mock_callback
        )

    def test_add_message_flow(self):
        """测试添加消息流程"""
        # Mock trigger: first False, then True
        self.mock_trigger_manager.should_trigger.side_effect = [
            (False, None),
            (True, FlushReason.MESSAGE_COUNT)
        ]

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 1. 添加第一条消息
        self.layer.perceive("user", "msg1", identity)

        # 验证 Buffer 状态
        buffer = self.layer.get_buffer(identity)
        assert buffer.message_count == 1

        # 验证未触发 Flush
        self.mock_callback.assert_not_called()

        # 2. 添加第二条消息 (触发 Flush)
        self.layer.perceive("assistant", "msg2", identity)

        # 验证 Flush 被调用
        self.mock_callback.assert_called_once()
        args, _ = self.mock_callback.call_args
        messages, reason = args
        assert len(messages) == 2
        assert reason == FlushReason.MESSAGE_COUNT

        # 验证 Buffer 被清空
        assert buffer.message_count == 0

    def test_flush_buffer_manual(self):
        """测试手动 Flush"""
        self.mock_trigger_manager.should_trigger.return_value = (False, None)

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        self.layer.perceive("user", "msg1", identity)

        self.layer.flush_buffer(identity)

        self.mock_callback.assert_called_once()
        args, _ = self.mock_callback.call_args
        messages, reason = args
        assert len(messages) == 1
        assert reason == FlushReason.MANUAL


class TestSemanticFlowPerceptionLayer:
    """测试语义流感知层"""

    def setup_method(self):
        self.mock_parser = Mock()
        self.mock_adsorber = Mock()
        self.mock_relay = Mock()
        self.mock_callback = Mock()

        self.config = SemanticFlowPerceptionConfig()

        # v3.0: should_adsorb 返回 None 表示继续吸附
        self.mock_adsorber.should_adsorb.return_value = None
        self.mock_adsorber.compute_new_topic_kernel.return_value = [0.1, 0.2, 0.3]

        # v3.0: should_relay 返回 None 表示不需要接力
        self.mock_relay.should_relay.return_value = None

        self.layer = SemanticFlowPerceptionLayer(
            config=self.config,
            parser=self.mock_parser,
            adsorber=self.mock_adsorber,
            relay_controller=self.mock_relay,
            on_flush_callback=self.mock_callback
        )

    def test_process_new_block_flow(self):
        """测试新 Block 处理流程"""
        user_msg = StreamMessage(message_type=StreamMessageType.USER, content="hi")
        assistant_msg = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="hello")

        self.mock_parser.parse_message.side_effect = [user_msg, assistant_msg]

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 添加用户消息
        self.layer.perceive("user", "hi", identity)

        # 添加助手消息完成 block
        self.layer.perceive("assistant", "hello", identity)

        # Verify: block 应该已完成并加入 buffer
        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0].is_complete

        # 验证调用了吸附器计算核心
        self.mock_adsorber.compute_new_topic_kernel.assert_called()

    def test_semantic_drift_flush(self):
        """测试语义漂移触发 Flush"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 第一轮对话
        msg1_user = StreamMessage(message_type=StreamMessageType.USER, content="old topic")
        msg1_assistant = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="old response")

        # 第二轮对话
        msg2_user = StreamMessage(message_type=StreamMessageType.USER, content="new topic")
        msg2_assistant = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="new response")

        self.mock_parser.parse_message.side_effect = [
            msg1_user, msg1_assistant,
            msg2_user, msg2_assistant
        ]

        # 第一轮：正常吸附 (返回 None)
        self.mock_adsorber.should_adsorb.return_value = None

        self.layer.perceive("user", "old topic", identity)
        self.layer.perceive("assistant", "old response", identity)

        # 验证第一个 block 已加入
        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 第二轮：语义漂移 (返回 FlushEvent)
        # 创建一个包含第一个 block 的 FlushEvent
        first_block = buffer.blocks[0]
        flush_event = FlushEvent(
            flush_reason=FlushReason.SEMANTIC_DRIFT,
            blocks_to_flush=[first_block],
        )
        self.mock_adsorber.should_adsorb.return_value = flush_event

        self.layer.perceive("user", "new topic", identity)
        self.layer.perceive("assistant", "new response", identity)

        # 验证 Flush 被调用
        self.mock_callback.assert_called()

        # 验证 Buffer 只有新 block（旧的已被 flush）
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0].user_block.content == "new topic"

    def test_token_overflow_relay(self):
        """测试 Token 溢出接力"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 第一轮对话
        msg1_user = StreamMessage(message_type=StreamMessageType.USER, content="first")
        msg1_assistant = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="response1")

        # 第二轮对话
        msg2_user = StreamMessage(message_type=StreamMessageType.USER, content="second")
        msg2_assistant = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="response2")

        self.mock_parser.parse_message.side_effect = [
            msg1_user, msg1_assistant,
            msg2_user, msg2_assistant
        ]

        # 第一轮：正常吸附
        self.mock_adsorber.should_adsorb.return_value = None
        self.mock_relay.should_relay.return_value = None

        self.layer.perceive("user", "first", identity)
        self.layer.perceive("assistant", "response1", identity)

        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 第二轮：Token 溢出 (返回 FlushEvent)
        first_block = buffer.blocks[0]
        flush_event = FlushEvent(
            flush_reason=FlushReason.TOKEN_OVERFLOW,
            blocks_to_flush=[first_block],
            relay_summary="Summary of previous conversation",
        )
        self.mock_relay.should_relay.return_value = flush_event

        self.layer.perceive("user", "second", identity)
        self.layer.perceive("assistant", "response2", identity)

        # 验证 Flush 被调用
        self.mock_callback.assert_called()

        # 验证 Relay Summary 被设置
        assert buffer.relay_summary == "Summary of previous conversation"

    def test_no_flush_callback_when_no_blocks(self):
        """测试无 blocks 时不调用回调"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 手动 flush 空 buffer
        result = self.layer.flush_buffer(identity)

        assert result == []
        self.mock_callback.assert_not_called()

    def test_clear_buffer(self):
        """测试清理 buffer"""
        user_msg = StreamMessage(message_type=StreamMessageType.USER, content="hi")
        assistant_msg = StreamMessage(message_type=StreamMessageType.ASSISTANT, content="hello")

        self.mock_parser.parse_message.side_effect = [user_msg, assistant_msg]

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        self.layer.perceive("user", "hi", identity)
        self.layer.perceive("assistant", "hello", identity)

        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 清理
        result = self.layer.clear_buffer(identity)
        assert result is True

        # 验证 buffer 已清空
        assert len(buffer.blocks) == 0
        assert buffer.topic_kernel_vector is None
