"""
价值评估器 (ValueGater) 单元测试

测试覆盖:
- RuleBasedGater: 规则引擎评估
- 白名单关键词检测
- 黑名单过滤
- 长度过滤
- 实质内容提取
"""

import pytest
from hivememory.core.models import ConversationMessage
from hivememory.generation.gating import RuleBasedGater


class TestRuleBasedGater:
    """测试规则引擎评估器"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.gater = RuleBasedGater(
            min_total_length=20,
            min_substantive_length=10
        )

    # ========== 白名单测试 ==========

    def test_valuable_code_snippet(self):
        """测试代码片段被判定为有价值"""
        messages = [
            ConversationMessage(
                role="user",
                content="帮我写一个快排算法",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="好的，这是 Python 实现：\n```python\ndef quicksort(arr): ...\n```",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is True

    def test_valuable_configuration(self):
        """测试配置相关内容被判定为有价值"""
        messages = [
            ConversationMessage(
                role="user",
                content="如何配置环境变量？",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="在 .env 文件中添加 API_KEY=xxx",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is True

    def test_valuable_error_fixing(self):
        """测试错误修复内容被判定为有价值"""
        messages = [
            ConversationMessage(
                role="user",
                content="遇到 TypeError 错误，如何修复？",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="这个 bug 可以通过类型检查解决",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is True

    # ========== 黑名单测试 ==========

    def test_trivial_greeting(self):
        """测试简单寒暄被过滤"""
        messages = [
            ConversationMessage(
                role="user",
                content="你好",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="你好！有什么可以帮助你的吗？",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is False

    def test_trivial_thanks(self):
        """测试简单感谢被过滤"""
        messages = [
            ConversationMessage(
                role="user",
                content="谢谢",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="不客气！",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is False

    def test_trivial_confirmation(self):
        """测试简单确认被过滤"""
        messages = [
            ConversationMessage(
                role="user",
                content="好的，明白了",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is False

    # ========== 长度过滤测试 ==========

    def test_too_short_conversation(self):
        """测试过短对话被过滤"""
        messages = [
            ConversationMessage(
                role="user",
                content="嗯",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is False

    def test_sufficient_length(self):
        """测试足够长度的对话通过"""
        messages = [
            ConversationMessage(
                role="user",
                content="我想了解一下 Python 中的装饰器是如何工作的",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        assert self.gater.evaluate(messages) is True

    # ========== 边界情况测试 ==========

    def test_empty_messages(self):
        """测试空消息列表"""
        assert self.gater.evaluate([]) is False

    def test_mixed_content(self):
        """测试混合内容（寒暄 + 实质内容）"""
        messages = [
            ConversationMessage(
                role="user",
                content="你好，帮我写一个函数",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="好的，这是函数实现...",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        # 包含实质内容，应该通过
        assert self.gater.evaluate(messages) is True

    def test_contains_valuable_keywords(self):
        """测试白名单关键词检测"""
        # 测试代码关键词
        assert self.gater._contains_valuable_keywords("这是一个 function 实现") is True
        assert self.gater._contains_valuable_keywords("import numpy") is True
        assert self.gater._contains_valuable_keywords("定义一个 class") is True

        # 测试配置关键词
        assert self.gater._contains_valuable_keywords("修改 config 文件") is True
        assert self.gater._contains_valuable_keywords("设置环境变量") is True

        # 测试技术关键词
        assert self.gater._contains_valuable_keywords("数据库查询") is True
        assert self.gater._contains_valuable_keywords("算法优化") is True

        # 测试无关键词
        assert self.gater._contains_valuable_keywords("随便聊聊天") is False

    def test_is_pure_trivial(self):
        """测试纯寒暄检测"""
        # 纯寒暄
        assert self.gater._is_pure_trivial("你好") is True
        assert self.gater._is_pure_trivial("谢谢你") is True
        assert self.gater._is_pure_trivial("好的，明白") is True

        # 包含实质内容
        assert self.gater._is_pure_trivial("你好，请帮我写代码") is False
        assert self.gater._is_pure_trivial("谢谢，这个函数很有用") is False

    def test_extract_substantive_content(self):
        """测试实质内容提取"""
        # 移除寒暄后的内容
        text = "你好，请帮我实现一个快排算法"
        cleaned = self.gater._extract_substantive_content(text)
        assert "实现" in cleaned
        assert "快排" in cleaned
        assert len(cleaned) > 10

        # 纯寒暄内容
        text = "你好，谢谢"
        cleaned = self.gater._extract_substantive_content(text)
        assert len(cleaned) < 5


class TestGaterIntegration:
    """集成测试 - 真实场景"""

    def setup_method(self):
        self.gater = RuleBasedGater()

    def test_real_code_conversation(self):
        """真实代码对话场景"""
        messages = [
            ConversationMessage(
                role="user",
                content="帮我写一个 Python 函数来解析 ISO8601 日期",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="""好的，这是实现：
```python
from datetime import datetime

def parse_iso8601(date_str):
    return datetime.fromisoformat(date_str)
```
""",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="user",
                content="太好了，谢谢！",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        # 包含代码，应该有价值
        assert self.gater.evaluate(messages) is True

    def test_real_idle_chat(self):
        """真实闲聊场景"""
        messages = [
            ConversationMessage(
                role="user",
                content="你好",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="你好！有什么可以帮助你的吗？",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="user",
                content="没事，随便聊聊",
                user_id="test_user",
                session_id="test_session"
            ),
            ConversationMessage(
                role="assistant",
                content="好的，很高兴和你聊天！",
                user_id="test_user",
                session_id="test_session"
            )
        ]

        # 纯闲聊，应该无价值
        assert self.gater.evaluate(messages) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
