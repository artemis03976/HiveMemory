"""
RetrievalRouter 单元测试

测试覆盖:
- SimpleRouter (基于规则)
- LLMRouter (Mock LLM 调用)
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from hivememory.generation.models import ConversationMessage
from hivememory.retrieval.router import SimpleRouter, LLMRouter, AlwaysRetrieveRouter, NeverRetrieveRouter

class TestSimpleRouter:
    """测试简单路由器"""

    def setup_method(self):
        self.router = SimpleRouter()

    def test_should_retrieve_keywords(self):
        """测试关键词触发"""
        assert self.router.should_retrieve("记得之前说过什么") is True
        assert self.router.should_retrieve("那个代码怎么写的") is True
        assert self.router.should_retrieve("我的 API Key 是多少") is True

    def test_should_not_retrieve_greetings(self):
        """测试问候语不触发"""
        assert self.router.should_retrieve("你好") is False
        assert self.router.should_retrieve("Hello there") is False
        assert self.router.should_retrieve("谢谢") is False

    def test_should_not_retrieve_creation(self):
        """测试创建类指令不触发"""
        assert self.router.should_retrieve("帮我写一个 Python 脚本") is False
        assert self.router.should_retrieve("生成一篇文章") is False

    def test_short_query_handling(self):
        """测试短查询处理"""
        assert self.router.should_retrieve("hi") is False
        # 短查询如果有上下文引用，应该触发
        assert self.router.should_retrieve("那个呢") is True

    def test_long_query_fallback(self):
        """测试长查询默认触发"""
        long_query = "这是一个非常长的查询，包含了大量的细节和上下文信息，通常这种情况下我们需要检索历史记忆来更好地回答用户的问题。"
        assert self.router.should_retrieve(long_query) is True


class TestLLMRouter:
    """测试 LLM 路由器"""

    def setup_method(self):
        self.mock_config = {"model": "gpt-4o-mini", "api_key": "test"}
        self.router = LLMRouter(llm_config=self.mock_config)

    def test_should_retrieve_yes(self):
        """测试 LLM 返回 YES"""
        mock_litellm = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "YES"
        mock_litellm.completion.return_value = mock_response

        with patch.dict(sys.modules, {'litellm': mock_litellm}):
            assert self.router.should_retrieve("What did we discuss yesterday?") is True

    def test_should_retrieve_no(self):
        """测试 LLM 返回 NO"""
        mock_litellm = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "NO"
        mock_litellm.completion.return_value = mock_response

        with patch.dict(sys.modules, {'litellm': mock_litellm}):
            assert self.router.should_retrieve("Write a poem about spring") is False

    def test_llm_failure_fallback(self):
        """测试 LLM 失败时回退到 SimpleRouter"""
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = Exception("API Error")
        
        with patch.dict(sys.modules, {'litellm': mock_litellm}):
            # SimpleRouter 应该能识别这个关键词
            assert self.router.should_retrieve("记得之前") is True


class TestUtilityRouters:
    """测试工具路由器"""

    def test_always_retrieve(self):
        router = AlwaysRetrieveRouter()
        assert router.should_retrieve("any query") is True
        assert router.should_retrieve("") is True

    def test_never_retrieve(self):
        router = NeverRetrieveRouter()
        assert router.should_retrieve("remember this?") is False
        assert router.should_retrieve("") is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
