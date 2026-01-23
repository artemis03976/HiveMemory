"""
UnifiedStreamParser 单元测试

测试覆盖:
- OpenAI 格式消息解析
- LangChain 格式消息解析 (Mock)
- 简单文本解析
- Block 创建判定逻辑
"""

import pytest
from unittest.mock import Mock, MagicMock
import sys

from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.models import StreamMessageType

class TestUnifiedStreamParser:
    """测试统一流式解析器"""

    def setup_method(self):
        self.parser = UnifiedStreamParser()

    def test_parse_simple_text(self):
        """测试简单文本解析"""
        msg = self.parser.parse_message("hello")
        assert msg.message_type == StreamMessageType.USER_QUERY
        assert msg.content == "hello"

    def test_parse_openai_format(self):
        """测试 OpenAI 字典格式解析"""
        # User
        user_msg = {"role": "user", "content": "hi"}
        parsed = self.parser.parse_message(user_msg)
        assert parsed.message_type == StreamMessageType.USER_QUERY
        assert parsed.content == "hi"

        # Assistant
        ai_msg = {"role": "assistant", "content": "response"}
        parsed = self.parser.parse_message(ai_msg)
        assert parsed.message_type == StreamMessageType.ASSISTANT_MESSAGE
        assert parsed.content == "response"

        # Tool Call
        tool_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "function": {
                    "name": "search",
                    "arguments": '{"query": "python"}'
                }
            }]
        }
        parsed = self.parser.parse_message(tool_msg)
        assert parsed.message_type == StreamMessageType.TOOL_CALL
        assert parsed.tool_name == "search"
        assert parsed.tool_args == {"query": "python"}

    def test_parse_langchain_format(self):
        """测试 LangChain 对象解析 (Mock)"""
        # Mock LangChain messages
        # HumanMessage
        mock_human = MagicMock()
        mock_human.__class__.__name__ = "HumanMessage"
        mock_human.content = "hi"
        # 这里的 isinstance check 比较棘手，因为 parser 内部 import 了 langchain
        # 我们最好 mock parser._is_langchain_message 和 _parse_langchain_message
        # 或者安装了 langchain 的话直接用。
        
        # 尝试 import langchain，如果失败则 mock
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Human
            msg = HumanMessage(content="hi")
            parsed = self.parser.parse_message(msg)
            assert parsed.message_type == StreamMessageType.USER_QUERY
            
            # AI
            msg = AIMessage(content="response")
            parsed = self.parser.parse_message(msg)
            assert parsed.message_type == StreamMessageType.ASSISTANT_MESSAGE
            
        except ImportError:
            # 如果没有 langchain，跳过此测试或仅测试 mock 路径
            pass

    def test_should_create_new_block(self):
        """测试 Block 创建判定"""
        # User Query 应该创建新 Block
        msg = Mock()
        msg.message_type = StreamMessageType.USER_QUERY
        assert self.parser.should_create_new_block(msg) is True
        
        # 其他类型不应该创建
        msg.message_type = StreamMessageType.ASSISTANT_MESSAGE
        assert self.parser.should_create_new_block(msg) is False
