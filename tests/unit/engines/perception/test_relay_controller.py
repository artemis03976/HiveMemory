"""
RelayController 单元测试

测试覆盖:
- Token 溢出检测逻辑
- 摘要生成逻辑 (简单规则)
- 摘要上下文注入格式

Note:
    v3.0 重构：should_trigger_relay() 改为 should_relay()，返回 Optional[FlushEvent]
"""

import pytest
from unittest.mock import Mock

from hivememory.core.models import Identity
from hivememory.engines.perception.relay_controller import RelayController
from hivememory.engines.perception.models import (
    FlushEvent,
    LogicalBlock,
    SemanticBuffer,
    Triplet,
    FlushReason,
)
from hivememory.core.models import StreamMessage, StreamMessageType


class TestRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = RelayController(max_processing_tokens=100)

    def test_should_relay_no_overflow(self):
        """测试无溢出时返回 None"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.total_tokens = 80

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="Hello"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="Hi"),
            total_tokens=10,
        )

        # 80 + 10 <= 100 -> None (no overflow)
        result = self.controller.should_relay(buffer, new_block)
        assert result is None

    def test_should_relay_with_overflow(self):
        """测试溢出时返回 FlushEvent"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.total_tokens = 80

        # 添加一个 block 到 buffer
        existing_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="Existing"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="Response"),
            total_tokens=80,
        )
        buffer.blocks.append(existing_block)

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="New"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="Response"),
            total_tokens=30,
        )

        # 80 + 30 > 100 -> FlushEvent
        result = self.controller.should_relay(buffer, new_block)

        assert result is not None
        assert isinstance(result, FlushEvent)
        assert result.flush_reason == FlushReason.TOKEN_OVERFLOW
        assert result.relay_summary is not None
        assert result.triggered_by_block is new_block
        assert len(result.blocks_to_flush) == 1

    def test_generate_simple_summary(self):
        """测试简单摘要生成"""
        # 构造 Block 链
        block1 = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="查询天气"
            )
        )

        block2 = LogicalBlock(
            execution_chain=[
                Triplet(
                    tool_name="weather_api",
                    observation="晴天"
                )
            ]
        )

        summary = self.controller.generate_summary([block1, block2])

        assert "处理了 1 个用户请求" in summary
        assert "weather_api" in summary
        assert "查询天气" in summary

    def test_create_relay_context(self):
        """测试上下文注入格式"""
        summary = "Test Summary"
        context = self.controller.create_relay_context(summary)
        assert "[接力摘要]" in context
        assert "Test Summary" in context

    def test_flush_event_contains_relay_summary(self):
        """测试 FlushEvent 包含接力摘要"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.total_tokens = 90

        # 添加 blocks
        block1 = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="查询1"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复1"),
            total_tokens=45,
        )
        block2 = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="查询2"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复2"),
            total_tokens=45,
        )
        buffer.blocks.extend([block1, block2])

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="新查询"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="新回复"),
            total_tokens=20,
        )

        result = self.controller.should_relay(buffer, new_block)

        assert result is not None
        assert result.relay_summary is not None
        assert "处理了 2 个用户请求" in result.relay_summary
