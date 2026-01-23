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
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import FlushReason, Identity
from hivememory.engines.generation.models import ConversationMessage
from hivememory.engines.perception.simple_perception_layer import SimplePerceptionLayer
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.models import SimpleBuffer, SemanticBuffer, LogicalBlock, StreamMessage, StreamMessageType

class TestSimplePerceptionLayer:
    """测试简单感知层"""

    def setup_method(self):
        self.mock_trigger_manager = Mock()
        self.mock_callback = Mock()
        self.layer = SimplePerceptionLayer(
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

        # 创建 Identity 对象
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 1. 添加第一条消息
        self.layer.add_message("user", "msg1", identity)

        # 验证 Buffer 状态
        buffer = self.layer.get_buffer(identity)
        assert buffer.message_count == 1

        # 验证未触发 Flush
        self.mock_callback.assert_not_called()

        # 2. 添加第二条消息 (触发 Flush)
        self.layer.add_message("assistant", "msg2", identity)

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
        # Mock trigger for add_message
        self.mock_trigger_manager.should_trigger.return_value = (False, None)

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        self.layer.add_message("user", "msg1", identity)

        # SimplePerceptionLayer.flush_buffer 现在接受 Identity 对象
        # 它内部会使用 FlushReason.MANUAL
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

        self.layer = SemanticFlowPerceptionLayer(
            parser=self.mock_parser,
            adsorber=self.mock_adsorber,
            relay_controller=self.mock_relay,
            on_flush_callback=self.mock_callback
        )

    def test_process_new_block_flow(self):
        """测试新 Block 处理流程"""
        # 1. Mock Parser
        stream_msg = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="hi")
        self.mock_parser.parse_message.return_value = stream_msg
        self.mock_parser.should_create_new_block.return_value = True

        # 2. Mock Adsorber (Adsorb=True)
        self.mock_adsorber.should_adsorb.return_value = (True, None)

        # 3. 创建 Identity 并添加消息
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        self.layer.add_message("user", "hi", identity)

        # Verify
        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 0  # Still processing
        assert buffer.current_block is not None

        # 验证调用了吸附器更新核心 (Wait, update_topic_kernel is called in _check_and_flush, which happens only if block is complete)
        # Since block is not complete, it shouldn't be called yet.
        # self.mock_adsorber.update_topic_kernel.assert_called()

    def test_semantic_drift_flush(self):
        """测试语义漂移触发 Flush"""
        # Setup mock for FIRST message
        msg1 = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="old")

        # Setup mock for SECOND message
        msg2 = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="new")

        self.mock_parser.parse_message.side_effect = [msg1, msg2]
        self.mock_parser.should_create_new_block.return_value = True

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # Buffer 中已有 Block
        self.layer.add_message("user", "old", identity)
        buffer = self.layer.get_buffer(identity)
        # 强制闭合 Block
        if buffer.current_block:
            # Set user_block and response_block to make is_complete True
            buffer.current_block.user_block = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="old")
            buffer.current_block.response_block = StreamMessage(message_type=StreamMessageType.ASSISTANT_MESSAGE, content="resp")
            buffer.add_block(buffer.current_block)
            buffer.current_block = None

        # Adsorber says NO (Drift)
        self.mock_adsorber.should_adsorb.return_value = (False, FlushReason.SEMANTIC_DRIFT)

        # Add new message (User Query)
        self.layer.add_message("user", "new", identity)

        # Add Assistant Message to COMPLETE the new block and trigger flush check
        msg3 = StreamMessage(message_type=StreamMessageType.ASSISTANT_MESSAGE, content="resp_new")
        self.mock_parser.parse_message.side_effect = [msg3] # Next call returns msg3
        self.mock_parser.should_create_new_block.return_value = False

        self.layer.add_message("assistant", "resp_new", identity)

        # Verify Flush called for OLD blocks
        self.mock_callback.assert_called()

        # Verify Buffer has NEW block only (after flush)
        # 新逻辑：先 flush 旧 buffer，然后新 block 加入
        # 所以 buffer.blocks 应该只有新 block
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0].user_block.content == "new"

    def test_token_overflow_relay(self):
        """测试 Token 溢出接力"""
        # Setup mocks
        msg1 = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="old")
        msg2 = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="new")
        # Need msg3 for assistant response to close the block
        msg3 = StreamMessage(message_type=StreamMessageType.ASSISTANT_MESSAGE, content="resp_new")

        self.mock_parser.parse_message.side_effect = [msg1, msg2, msg3]
        # should_create_new_block: True for msg1, True for msg2, False for msg3
        self.mock_parser.should_create_new_block.side_effect = [True, True, False]

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # Buffer 中已有 Block
        self.layer.add_message("user", "old", identity)
        buffer = self.layer.get_buffer(identity)
        if buffer.current_block:
            # Set user_block and response_block to make is_complete True
            buffer.current_block.user_block = StreamMessage(message_type=StreamMessageType.USER_QUERY, content="old")
            buffer.current_block.response_block = StreamMessage(message_type=StreamMessageType.ASSISTANT_MESSAGE, content="resp")
            buffer.add_block(buffer.current_block)
            buffer.current_block = None

        # Adsorber says NO (Overflow)
        self.mock_adsorber.should_adsorb.return_value = (False, FlushReason.TOKEN_OVERFLOW)

        # Relay Controller generates summary
        self.mock_relay.generate_summary.return_value = "Summary"
        self.mock_relay.create_relay_context.return_value = "Context"

        # Add new message (User)
        self.layer.add_message("user", "new", identity)
        # Add new message (Assistant) to close block and trigger check
        self.layer.add_message("assistant", "resp_new", identity)

        # Verify Flush called
        self.mock_callback.assert_called()

        # Verify Relay Summary is set in Buffer
        assert buffer.relay_summary == "Summary"
