"""
记忆提取器 (LLMMemoryExtractor) 单元测试

测试覆盖:
- LLM 消息格式转换
- JSON 输出解析 (多种格式容错)
- 提取流程逻辑
- 错误处理与重试机制
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
from datetime import datetime

from hivememory.engines.generation.models import ConversationMessage
from hivememory.patchouli.config import LLMConfig
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft


class TestLLMMemoryExtractor:
    """测试 LLM 记忆提取器"""

    def setup_method(self):
        """每个测试方法前执行"""
        # 使用真实的 Pydantic 模型作为基础，并进行 Mock
        self.mock_config = LLMConfig(
            model="test-model",
            api_key="test-key",
            api_base="https://api.test.com",
            temperature=0.0,
            max_tokens=1000
        )
        
        self.extractor = LLMMemoryExtractor(llm_config=self.mock_config)

    def test_convert_to_litellm_messages(self, mock_env):
        """测试 LangChain 消息转 LiteLLM 格式"""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        lc_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="user input"),
            AIMessage(content="assistant reply")
        ]
        
        litellm_msgs = self.extractor._convert_to_litellm_messages(lc_messages)
        
        assert len(litellm_msgs) == 3
        assert litellm_msgs[0] == {"role": "system", "content": "sys"}
        assert litellm_msgs[1] == {"role": "user", "content": "user input"}
        assert litellm_msgs[2] == {"role": "assistant", "content": "assistant reply"}

    def test_parse_json_pure(self):
        """测试解析纯 JSON"""
        json_str = json.dumps({
            "title": "Test Title",
            "summary": "Test Summary",
            "tags": ["tag1", "tag2"],
            "memory_type": "FACT",
            "content": "Test Content",
            "confidence_score": 0.9,
            "has_value": True
        })
        
        draft = self.extractor._parse_json_output(json_str)
        assert draft is not None
        assert draft.title == "Test Title"
        assert draft.confidence_score == 0.9

    def test_parse_json_markdown_block(self):
        """测试解析 Markdown 代码块中的 JSON"""
        json_str = """
        Here is the JSON:
        ```json
        {
            "title": "Block Title",
            "summary": "Block Summary",
            "tags": ["tag1"],
            "memory_type": "CODE_SNIPPET",
            "content": "code...",
            "confidence_score": 0.8,
            "has_value": true
        }
        ```
        """
        
        draft = self.extractor._parse_json_output(json_str)
        assert draft is not None
        assert draft.title == "Block Title"
        assert draft.memory_type == "CODE_SNIPPET"

    def test_parse_json_regex_fallback(self):
        """测试正则提取 JSON"""
        json_str = """
        Some text before
        {
            "title": "Regex Title",
            "summary": "Regex Summary",
            "tags": [],
            "memory_type": "FACT",
            "content": "content",
            "confidence_score": 0.5,
            "has_value": false
        }
        Some text after
        """
        
        draft = self.extractor._parse_json_output(json_str)
        assert draft is not None
        assert draft.title == "Regex Title"

    def test_parse_invalid_json(self):
        """测试解析无效 JSON"""
        draft = self.extractor._parse_json_output("Not a JSON string")
        assert draft is None

    @patch("hivememory.infrastructure.llm.litellm_service.litellm.completion")
    def test_extract_success(self, mock_completion):
        """测试成功提取流程"""
        # 模拟 LLM 响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Extracted",
            "summary": "Summary",
            "tags": ["t1"],
            "memory_type": "FACT",
            "content": "Content",
            "confidence_score": 0.95,
            "has_value": True
        })
        mock_completion.return_value = mock_response

        transcript = "User: Hi\nAssistant: Hello"
        metadata = {"user_id": "u1", "session_id": "s1"}

        draft = self.extractor.extract(transcript, metadata)

        assert draft is not None
        assert draft.title == "Extracted"
        mock_completion.assert_called_once()

    @patch("hivememory.infrastructure.llm.litellm_service.litellm.completion")
    def test_extract_retry_logic(self, mock_completion):
        """测试重试逻辑"""
        # 前几次抛出异常，最后一次成功
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Retry Success",
            "summary": "Summary",
            "tags": ["t1"],
            "memory_type": "FACT",
            "content": "Content",
            "confidence_score": 0.9,
            "has_value": True
        })

        mock_completion.side_effect = [Exception("Fail 1"), mock_response]

        transcript = "User: Hi"
        metadata = {}

        draft = self.extractor.extract(transcript, metadata)

        assert draft is not None
        assert draft.title == "Retry Success"
        assert mock_completion.call_count == 2

    @patch("hivememory.infrastructure.llm.litellm_service.litellm.completion")
    def test_extract_all_retries_fail(self, mock_completion):
        """测试所有重试均失败"""
        mock_completion.side_effect = Exception("Always Fail")

        transcript = "User: Hi"
        metadata = {}

        # 应该返回 None，并记录错误
        draft = self.extractor.extract(transcript, metadata)

        assert draft is None
        assert mock_completion.call_count == self.extractor.max_retries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
