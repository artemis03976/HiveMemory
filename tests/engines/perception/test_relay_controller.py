"""
TokenOverflowRelayController 单元测试

测试覆盖:
- Token 溢出检测逻辑
- 摘要生成逻辑 (简单规则)
- 摘要上下文注入格式
"""

import pytest
from unittest.mock import Mock

from hivememory.engines.perception.relay_controller import TokenOverflowRelayController
from hivememory.engines.perception.models import LogicalBlock, SemanticBuffer, StreamMessage, StreamMessageType

class TestTokenOverflowRelayController:
    """测试 Token 溢出接力控制器"""

    def setup_method(self):
        self.controller = TokenOverflowRelayController(max_processing_tokens=100)

    def test_should_trigger_relay(self):
        """测试溢出检测"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 80
        
        new_block = Mock(spec=LogicalBlock)
        new_block.total_tokens = 10
        
        # 80 + 10 <= 100 -> False
        assert self.controller.should_trigger_relay(buffer, new_block) is False
        
        # 80 + 30 > 100 -> True
        new_block.total_tokens = 30
        assert self.controller.should_trigger_relay(buffer, new_block) is True

    def test_generate_simple_summary(self):
        """测试简单摘要生成"""
        # 构造 Block 链
        block1 = LogicalBlock()
        block1.add_stream_message(StreamMessage(
            message_type=StreamMessageType.USER,
            content="查询天气"
        ))
        
        block2 = LogicalBlock()
        block2.add_stream_message(StreamMessage(
            message_type=StreamMessageType.TOOL_CALL,
            content="call weather tool",
            tool_name="weather_api"
        ))
        
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
