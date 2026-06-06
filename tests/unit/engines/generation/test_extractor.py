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
from hivememory.system.config import LLMConfig, ExtractorConfig
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.i18n import set_default_language


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


class TestLLMMemoryExtractor:
    """测试 LLM 记忆提取器"""

    def setup_method(self):
        """每个测试方法前执行"""
        # 使用真实的 Pydantic 模型作为基础，并进行 Mock
        self.mock_llm_config = LLMConfig(
            model="test-model",
            api_key="test-key",
            api_base="https://api.test.com",
            temperature=0.0,
            max_tokens=1000
        )
        self.mock_service = Mock()
        self.mock_service.config = self.mock_llm_config
        
        self.extractor_config = ExtractorConfig()
        self.extractor = LLMMemoryExtractor(
            config=self.extractor_config,
            llm_service=self.mock_service
        )

    def test_extract_messages_are_litellm_format(self):
        """测试提取流程发送 LiteLLM 标准消息"""
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
        self.extractor.extract(transcript, metadata)

        _, kwargs = self.mock_service.complete_with_retry.call_args
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_default_language_selects_english_passive_prompts(self):
        """测试全局语言驱动被动提取模板"""
        set_default_language("en")
        extractor = LLMMemoryExtractor(
            config=ExtractorConfig(),
            llm_service=self.mock_service,
        )
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

        extractor.extract("User: Hi", {})

        _, kwargs = self.mock_service.complete_with_retry.call_args
        messages = kwargs["messages"]
        assert "memory manager" in messages[0]["content"]
        assert "Conversation" in messages[1]["content"]

    def test_default_language_selects_english_write_prompts(self):
        """测试全局语言驱动 WRITE 模板"""
        set_default_language("en")
        extractor = LLMMemoryExtractor(
            config=ExtractorConfig(),
            llm_service=self.mock_service,
        )
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

        extractor.extract(
            "User: Save this",
            {
                "mode": "write",
                "write_content": "Content to save",
            },
        )

        _, kwargs = self.mock_service.complete_with_retry.call_args
        messages = kwargs["messages"]
        assert "Active Response Mode" in messages[0]["content"]
        assert "Agent-submitted Memory Draft" in messages[1]["content"]
        assert "(Not provided)" in messages[1]["content"]

    def test_default_language_selects_english_update_prompts(self):
        """测试全局语言驱动 UPDATE 模板"""
        set_default_language("en")
        extractor = LLMMemoryExtractor(
            config=ExtractorConfig(),
            llm_service=self.mock_service,
        )
        json_output = json.dumps({
            "new_content": "New content",
            "changelog": "Updated content"
        })
        self.mock_service.complete_with_retry.return_value = json_output

        extractor.merge(
            old_content="Old content",
            metadata={
                "instruction": "Update it",
                "memory_title": "Title",
                "memory_alias": "alias",
            },
        )

        _, kwargs = self.mock_service.complete_with_retry.call_args
        messages = kwargs["messages"]
        assert "Editor Mode" in messages[0]["content"]
        assert "Target Memory" in messages[1]["content"]
        assert "No new material" in messages[1]["content"]
        assert "No background conversation" in messages[1]["content"]

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
