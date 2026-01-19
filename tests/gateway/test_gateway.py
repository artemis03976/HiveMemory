"""
Global Gateway 单元测试

测试 Gateway 的各个组件：
- RuleInterceptor (L1 规则拦截器)
- Gateway 数据模型
- GatewayResult 回退机制
- GlobalGateway 主类 (需要 Mock LLM)

作者: HiveMemory Team
��本: 2.0
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from hivememory.core.config import GatewayConfig
from hivememory.gateway.models import (
    GatewayIntent,
    ContentPayload,
    MemorySignal,
    GatewayResult,
    InterceptorResult,
)
from hivememory.gateway.interceptors import RuleInterceptor
from hivememory.gateway.prompts import get_system_prompt


class TestGatewayModels:
    """测试 Gateway 数据模型"""

    def test_gateway_intent_enum(self):
        """测试意图枚举"""
        assert GatewayIntent.RAG.value == "RAG"
        assert GatewayIntent.CHAT.value == "CHAT"
        assert GatewayIntent.TOOL.value == "TOOL"
        assert GatewayIntent.SYSTEM.value == "SYSTEM"

    def test_content_payload(self):
        """测试内容载荷模型"""
        payload = ContentPayload(
            rewritten_query="如何部署贪吃蛇游戏",
            search_keywords=["贪吃蛇", "部署"],
            target_filters={"memory_type": "CODE_SNIPPET"}
        )
        assert payload.rewritten_query == "如何部署贪吃蛇游戏"
        assert len(payload.search_keywords) == 2
        assert payload.target_filters["memory_type"] == "CODE_SNIPPET"

    def test_content_payload_defaults(self):
        """测试内容载荷默认值"""
        payload = ContentPayload(rewritten_query="测试查询")
        assert payload.rewritten_query == "测试查询"
        assert payload.search_keywords == []
        assert payload.target_filters == {}

    def test_memory_signal(self):
        """测试记忆信号模型"""
        signal = MemorySignal(
            worth_saving=True,
            reason="技术问题具有长期参考价值"
        )
        assert signal.worth_saving is True
        assert "长期" in signal.reason

    def test_gateway_result(self):
        """测试 Gateway 结果模型"""
        result = GatewayResult(
            intent=GatewayIntent.RAG,
            content_payload=ContentPayload(
                rewritten_query="查询",
                search_keywords=["关键词"]
            ),
            memory_signal=MemorySignal(
                worth_saving=True,
                reason="测试"
            )
        )
        assert result.intent == GatewayIntent.RAG
        assert result.schema_version == "2.0"
        assert result.gateway_parse_failed is False

    def test_gateway_result_fallback(self):
        """测试 Gateway 结果回退"""
        result = GatewayResult.fallback("原始查询")
        assert result.intent == GatewayIntent.CHAT
        assert result.content_payload.rewritten_query == "原始查询"
        assert result.content_payload.search_keywords == []
        assert result.memory_signal.worth_saving is False
        assert result.gateway_parse_failed is True

    def test_interceptor_result(self):
        """测试拦截器结果模型"""
        result = InterceptorResult(
            intent=GatewayIntent.SYSTEM,
            reason="系统指令: /clear",
            hit=True
        )
        assert result.intent == GatewayIntent.SYSTEM
        assert result.reason == "系统指令: /clear"
        assert result.hit is True


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


class TestGatewayConfig:
    """测试 Gateway 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = GatewayConfig()
        assert config.enable_l1_interceptor is True
        assert config.enable_l2_semantic is True
        assert config.context_window == 3
        assert config.llm_temperature == 0.0
        assert config.llm_max_tokens == 512

    def test_custom_config(self):
        """测试自定义配置"""
        config = GatewayConfig(
            enable_l1_interceptor=False,
            context_window=5,
            llm_temperature=0.1
        )
        assert config.enable_l1_interceptor is False
        assert config.context_window == 5
        assert config.llm_temperature == 0.1


class TestSystemPrompts:
    """测试 System Prompt"""

    def test_get_default_prompt(self):
        """测试获取默认 Prompt"""
        prompt = get_system_prompt()
        assert "意图分类" in prompt
        assert "指代消解" in prompt
        assert "元数据提取" in prompt

    def test_get_simple_prompt(self):
        """测试获取简化版 Prompt"""
        prompt = get_system_prompt(variant="simple")
        assert len(prompt) < len(get_system_prompt())  # 简化版更短

    def test_get_english_prompt(self):
        """测试获取英文 Prompt"""
        prompt = get_system_prompt(language="en")
        assert "Intent Classification" in prompt


class TestGlobalGatewayIntegration:
    """测试 GlobalGateway 主类（需要 Mock）"""

    def test_init_with_config(self):
        """测试使用配置初始化"""
        mock_llm = Mock()
        mock_llm.model = "gpt-4o-mini"

        config = GatewayConfig(context_window=5)

        from hivememory.gateway import GlobalGateway

        gateway = GlobalGateway(llm_service=mock_llm, config=config)

        assert gateway.llm_service == mock_llm
        assert gateway.config.context_window == 5
        assert gateway.rule_interceptor is not None

    def test_init_without_l1_interceptor(self):
        """测试禁用 L1 拦截器"""
        mock_llm = Mock()
        mock_llm.model = "gpt-4o-mini"

        config = GatewayConfig(enable_l1_interceptor=False)

        from hivememory.gateway import GlobalGateway

        gateway = GlobalGateway(llm_service=mock_llm, config=config)

        assert gateway.rule_interceptor is None

    @patch('hivememory.gateway.gateway.GlobalGateway._semantic_analysis')
    def test_process_l1_intercept(self, mock_semantic):
        """测试 L1 拦截路径"""
        mock_llm = Mock()
        mock_llm.model = "test-model"

        from hivememory.gateway import GlobalGateway

        gateway = GlobalGateway(llm_service=mock_llm)

        # 测试系统指令拦截
        result = gateway.process("/clear")
        assert result.intent == GatewayIntent.SYSTEM
        assert result.model_used == "L1_Rule_Interceptor"
        # 不应调用 L2 语义分析
        mock_semantic.assert_not_called()

        # 测试闲聊拦截
        result = gateway.process("你好")
        assert result.intent == GatewayIntent.CHAT
        mock_semantic.assert_not_called()

    @patch('hivememory.gateway.gateway.GlobalGateway._semantic_analysis')
    def test_process_l2_semantic(self, mock_semantic):
        """测试 L2 语义分析路径"""
        mock_llm = Mock()
        mock_llm.model = "test-model"

        # Mock L2 返回结果
        mock_semantic.return_value = GatewayResult(
            intent=GatewayIntent.RAG,
            content_payload=ContentPayload(
                rewritten_query="如何部署贪吃蛇游戏",
                search_keywords=["贪吃蛇", "部署"]
            ),
            memory_signal=MemorySignal(
                worth_saving=True,
                reason="技术问题"
            )
        )

        from hivememory.gateway import GlobalGateway

        # 禁用 L1 拦截器以测试 L2
        config = GatewayConfig(enable_l1_interceptor=False)
        gateway = GlobalGateway(llm_service=mock_llm, config=config)

        result = gateway.process("怎么部署它？")

        assert result.intent == GatewayIntent.RAG
        assert "贪吃蛇" in result.content_payload.rewritten_query
        mock_semantic.assert_called_once()

    def test_semantic_analysis_parse_failure(self):
        """测试语义分析解析失败时的回退"""
        # Mock LLM 返回无效响应
        mock_llm = Mock()
        mock_llm.model = "test-model"

        # Mock 无效的响应（没有 tool_calls）
        mock_response = Mock()
        mock_response.choices = []
        mock_llm.complete_with_tools = Mock(return_value=mock_response)

        from hivememory.gateway import GlobalGateway

        config = GatewayConfig(enable_l1_interceptor=False)
        gateway = GlobalGateway(llm_service=mock_llm, config=config)

        result = gateway.process("测试查询")

        # 应该回退到保守策略
        assert result.gateway_parse_failed is True
        assert result.intent == GatewayIntent.CHAT
        assert result.content_payload.rewritten_query == "测试查询"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
