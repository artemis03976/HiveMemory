
import pytest
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.engines.gateway.interceptors import RuleInterceptor


class TestRuleInterceptor:
    """测试 L1 规则拦截器"""

    def test_system_command_intercept(self):
        """测试系统指令拦截"""
        interceptor = RuleInterceptor()

        # 测试各种系统指令
        for cmd in ["/clear", "/reset", "/start", "/help"]:
            result = interceptor.intercept(cmd)
            assert result is not None
            assert result.intent == GatewayIntent.SYSTEM
            assert result.hit is True

    def test_chat_intercept(self):
        """测试闲聊拦截"""
        interceptor = RuleInterceptor()

        # 测试各种闲聊
        for msg in ["你好", "hi", "谢谢", "thanks", "再见", "ok"]:
            result = interceptor.intercept(msg)
            assert result is not None
            assert result.intent == GatewayIntent.CHAT
            assert result.hit is True

    def test_no_intercept(self):
        """测试不拦截（需要 L2 处理）"""
        interceptor = RuleInterceptor()

        # 这些查询需要 L2 处理
        queries = [
            "如何部署贪吃蛇游戏",
            "Python 里怎么用 asyncio？",
            "我之前设置的 API Key 是什么？"
        ]

        for query in queries:
            result = interceptor.intercept(query)
            assert result is None  # 不拦截

    def test_empty_query(self):
        """测试空查询"""
        interceptor = RuleInterceptor()
        result = interceptor.intercept("   ")
        assert result is not None
        assert result.intent == GatewayIntent.CHAT

    def test_custom_patterns(self):
        """测试自定义模式"""
        custom_system = [r"^/custom$"]
        custom_chat = [r"^测试$"]

        interceptor = RuleInterceptor(
            custom_system_patterns=custom_system,
            custom_chat_patterns=custom_chat
        )

        result = interceptor.intercept("/custom")
        assert result.intent == GatewayIntent.SYSTEM

        result = interceptor.intercept("测试")
        assert result.intent == GatewayIntent.CHAT

    def test_add_pattern_dynamically(self):
        """测试动态添加模式"""
        interceptor = RuleInterceptor()

        # 添加自定义系统指令模式
        interceptor.add_system_pattern(r"^/restart$")

        result = interceptor.intercept("/restart")
        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM
