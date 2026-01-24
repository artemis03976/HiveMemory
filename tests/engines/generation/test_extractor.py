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

from hivememory.core.models import StreamMessage
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
        self.mock_service = Mock()
        self.mock_service.config = self.mock_config
        
        self.extractor = LLMMemoryExtractor(llm_service=self.mock_service)

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

    # test_parse_json_* tests removed as they test internal implementation details
    # or should be tested via parse_llm_json unit tests.

    def test_extract_success(self):
        """测试成功提取流程"""
        # 模拟 LLM 响应
        json_output = json.dumps({
            "title": "Extracted",
            "summary": "Summary",
            "tags": ["t1"],
            "memory_type": "FACT",
            "content": "Content",
            "confidence_score": 0.95,
            "has_value": True
        })
        self.mock_service.complete_with_retry.return_value = json_output

        transcript = "User: Hi\nAssistant: Hello"
        metadata = {"user_id": "u1", "session_id": "s1"}

        draft = self.extractor.extract(transcript, metadata)

        assert draft is not None
        assert draft.title == "Extracted"
        self.mock_service.complete_with_retry.assert_called_once()

    # test_extract_retry_logic removed as retry logic is handled by llm_service

    def test_extract_all_retries_fail(self):
        """测试 LLM 调用失败的情况"""
        # 模拟 complete_with_retry 抛出异常
        self.mock_service.complete_with_retry.side_effect = Exception("LLM Error")
        
        transcript = "User: Hi"
        metadata = {}
        
        # 应该返回 None，并记录错误
        draft = self.extractor.extract(transcript, metadata)
        
        assert draft is None
        self.mock_service.complete_with_retry.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
