from hivememory.engines.gateway.interceptors import RuleInterceptor
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.system.config import RuleInterceptorConfig
from hivememory.gateway.commands import (
    CommandCategory,
    CommandDefinition,
    CommandParseStatus,
    CommandRegistry,
    CommandRouteTarget,
    create_builtin_command_registry,
)


class TestRuleInterceptor:
    """测试 L1 规则拦截器。"""

    def test_system_command_intercept_with_injected_registry(self):
        """显式注入系统指令库时，已注册指令应解析为 MATCHED。"""
        interceptor = RuleInterceptor(
            config=RuleInterceptorConfig(),
            command_registry=create_builtin_command_registry(),
        )

        for command in ["/clear", "/reset", "/start", "/help"]:
            result = interceptor.intercept(command)
            assert result is not None
            assert result.intent == GatewayIntent.SYSTEM
            assert result.hit is True
            assert result.command is not None
            assert result.command.parse_status == CommandParseStatus.MATCHED

    def test_without_command_registry_filters_slash_command_as_unknown(self):
        """未注入系统指令库时，slash 输入仍应被 L1 过滤，避免进入 L2。"""
        interceptor = RuleInterceptor(config=RuleInterceptorConfig())

        result = interceptor.intercept("/clear")

        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM
        assert result.hit is True
        assert result.command is not None
        assert result.command.command_id is None
        assert result.command.parse_status == CommandParseStatus.UNKNOWN

    def test_chat_intercept(self):
        """测试闲聊拦截。"""
        interceptor = RuleInterceptor(config=RuleInterceptorConfig())

        for message in ["你好", "hi", "谢谢", "thanks", "再见", "ok"]:
            result = interceptor.intercept(message)
            assert result is not None
            assert result.intent == GatewayIntent.CHAT
            assert result.hit is True
            assert result.command is None

    def test_no_intercept(self):
        """普通非 slash 查询仍交给 L2。"""
        interceptor = RuleInterceptor(config=RuleInterceptorConfig())

        queries = [
            "如何部署贪吃蛇游戏",
            "Python 里怎么用 asyncio",
            "我之前设置的 API Key 是什么？",
        ]

        for query in queries:
            result = interceptor.intercept(query)
            assert result is None

    def test_empty_query(self):
        """空查询按简单聊天拦截。"""
        interceptor = RuleInterceptor(config=RuleInterceptorConfig())

        result = interceptor.intercept("   ")

        assert result is not None
        assert result.intent == GatewayIntent.CHAT

    def test_add_system_command_dynamically(self):
        """动态注册系统指令入口替代旧 regex 扩展。"""
        interceptor = RuleInterceptor(config=RuleInterceptorConfig())
        interceptor.add_system_command("/restart", command_id="system.restart")

        result = interceptor.intercept("/restart")

        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM
        assert result.command is not None
        assert result.command.command_id == "system.restart"
        assert result.command.parse_status == CommandParseStatus.MATCHED

    def test_unknown_slash_command_intercepts_with_unknown_command(self):
        """注入 registry 时，未知 slash 指令同样被过滤为 UNKNOWN。"""
        interceptor = RuleInterceptor(
            config=RuleInterceptorConfig(),
            command_registry=create_builtin_command_registry(),
        )

        result = interceptor.intercept("/does-not-exist")

        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM
        assert result.command is not None
        assert result.command.parse_status == CommandParseStatus.UNKNOWN

    def test_uses_injected_command_registry(self):
        """自定义 registry 应作为唯一的系统指令来源。"""
        registry = CommandRegistry()
        registry.register(
            CommandDefinition(
                command_id="system.custom",
                category=CommandCategory.SYSTEM,
                primary_name="/custom",
                summary="自定义指令",
                route_target=CommandRouteTarget(name="system.custom"),
            )
        )
        interceptor = RuleInterceptor(
            config=RuleInterceptorConfig(),
            command_registry=registry,
        )

        result = interceptor.intercept("/custom")

        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM
        assert result.command is not None
        assert result.command.command_id == "system.custom"


def test_disabled_system_command_matching_filters_slash_command_as_unknown():
    """显式关闭 system 解析后，slash 输入仍走 L1 兜底过滤。"""
    interceptor = RuleInterceptor(
        config=RuleInterceptorConfig(enable_system=False),
        command_registry=create_builtin_command_registry(),
    )

    result = interceptor.intercept("/help")

    assert result is not None
    assert result.intent == GatewayIntent.SYSTEM
    assert result.hit is True
    assert result.command is not None
    assert result.command.command_id is None
    assert result.command.parse_status == CommandParseStatus.UNKNOWN
