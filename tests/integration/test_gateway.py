"""
Gateway 单元测试

测试 Gateway 的各个组件：
- Gateway 数据模型
- GatewayResult 回退机制
- GatewayService

作者: HiveMemory Team
版本: 2.1
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from hivememory.patchouli.config import GatewayConfig
from hivememory.core.models import MemoryType, StreamMessage
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    SemanticAnalysisResult,
)
from hivememory.patchouli.protocol.models import QueryFilters
from hivememory.engines.gateway.interceptors import RuleInterceptor
from hivememory.engines.gateway.service import GatewayService
from hivememory.engines.gateway.prompts import get_system_prompt


class TestGatewayModels:
    """测试 Gateway 数据模型"""

    def test_gateway_intent_enum(self):
        """测试意图枚举"""
        assert GatewayIntent.RAG.value == "RAG"
        assert GatewayIntent.CHAT.value == "CHAT"
        assert GatewayIntent.TOOL.value == "TOOL"
        assert GatewayIntent.SYSTEM.value == "SYSTEM"

    def test_gateway_result(self):
        """测试 Gateway 结果模型"""
        result = GatewayResult(
            intent=GatewayIntent.RAG,
            rewritten_query="如何部署贪吃蛇游戏",
            search_keywords=["贪吃蛇", "部署"],
            worth_saving=True,
            reason="技术问题"
        )
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "如何部署贪吃蛇游戏"
        assert len(result.search_keywords) == 2
        assert result.worth_saving is True
        assert result.gateway_parse_failed is False

    def test_gateway_result_defaults(self):
        """测试 GatewayResult 默认值"""
        result = GatewayResult(
            intent=GatewayIntent.CHAT,
            rewritten_query="测试查询",
            worth_saving=False,
            reason="测试"
        )
        assert result.rewritten_query == "测试查询"
        assert result.search_keywords == []
        assert isinstance(result.target_filters, QueryFilters)
        assert result.target_filters.is_empty()
        assert result.processing_time_ms == 0.0
        assert result.model_used is None

    def test_gateway_result_fallback(self):
        """测试 Gateway 结果回退"""
        result = GatewayResult.fallback("原始查询")
        assert result.intent == GatewayIntent.CHAT
        assert result.rewritten_query == "原始查询"
        assert result.search_keywords == []
        assert result.worth_saving is False
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

    def test_semantic_analysis_result(self):
        """测试 L2 语义分析结果模型"""
        result = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="如何部署 Python 项目",
            search_keywords=["Python", "部署"],
            target_filters=QueryFilters(memory_type=MemoryType.CODE_SNIPPET),
            worth_saving=True,
            reason="技术问题具有长期参考价值",
            model="gpt-4o-mini"
        )
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "如何部署 Python 项目"
        assert len(result.search_keywords) == 2
        assert result.target_filters.memory_type == MemoryType.CODE_SNIPPET
        assert result.worth_saving is True
        assert result.model == "gpt-4o-mini"

    def test_is_l1_intercepted_property(self):
        """测试 is_l1_intercepted 属性"""
        result = GatewayResult(
            intent=GatewayIntent.SYSTEM,
            rewritten_query="/clear",
            worth_saving=False,
            reason="系统指令",
            l1_result=InterceptorResult(
                intent=GatewayIntent.SYSTEM,
                reason="系统指令",
                hit=True
            )
        )
        assert result.is_l1_intercepted is True

        result_no_l1 = GatewayResult(
            intent=GatewayIntent.CHAT,
            rewritten_query="你好",
            worth_saving=False,
            reason="闲聊"
        )
        assert result_no_l1.is_l1_intercepted is False


class TestGatewayService:
    """测试 GatewayService"""

    def test_init_with_interceptor(self):
        """测试使用拦截器初始化"""
        interceptor = RuleInterceptor()
        service = GatewayService(interceptor=interceptor)

        assert service.interceptor is not None
        assert service.semantic_analyzer is None

    def test_execute_l1(self):
        """测试 L1 拦截执行"""
        interceptor = RuleInterceptor()
        service = GatewayService(interceptor=interceptor)

        result = service.execute_l1("/clear")
        assert result is not None
        assert result.intent == GatewayIntent.SYSTEM

    def test_execute_l1_no_interceptor(self):
        """测试无拦截器时返回 None"""
        service = GatewayService(interceptor=None)

        result = service.execute_l1("/clear")
        assert result is None

    def test_process_l1_hit(self):
        """测试 L1 命中路径"""
        interceptor = RuleInterceptor()
        service = GatewayService(interceptor=interceptor)

        result = service.process("你好")

        assert result.intent == GatewayIntent.CHAT
        assert result.rewritten_query == "你好"
        assert result.is_l1_intercepted is True

    def test_process_l1_no_hit_no_l2(self):
        """测试 L1 未命中且 L2 禁用"""
        service = GatewayService(interceptor=None, semantic_analyzer=None)

        result = service.process("如何部署项目？")

        assert result.intent == GatewayIntent.CHAT
        assert result.rewritten_query == "如何部署项目？"
        assert result.search_keywords == []

    def test_process_with_mock_l2(self):
        """测试带 Mock L2 的完整流程"""
        interceptor = RuleInterceptor()

        # Mock L2 分析器
        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="如何部署 Python 项目",
            search_keywords=["Python", "部署"],
            target_filters={},
            worth_saving=True,
            reason="技术问题"
        ))

        service = GatewayService(
            interceptor=interceptor,
            semantic_analyzer=mock_analyzer
        )

        # L1 不会拦截这个查询，会走 L2
        result = service.process("怎么部署它？", context=[
            StreamMessage(message_type="user", content="我有一个 Python 项目"),
            StreamMessage(message_type="assistant", content="好的，告诉我更多")
        ])

        assert result.intent == GatewayIntent.RAG
        assert "Python" in result.rewritten_query
        assert result.search_keywords == ["Python", "部署"]
        assert result.worth_saving is True
        assert result.is_l1_intercepted is False


class TestGatewayConfig:
    """测试 Gateway 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = GatewayConfig()
        assert config.enable_l1_interceptor is True
        assert config.enable_l2_semantic is True
        assert config.context_window == 3
        assert config.prompt_variant == "default"
        assert config.prompt_language == "zh"

    def test_custom_config(self):
        """测试自定义配置"""
        config = GatewayConfig(
            enable_l1_interceptor=False,
            context_window=5,
            prompt_variant="simple"
        )
        assert config.enable_l1_interceptor is False
        assert config.context_window == 5
        assert config.prompt_variant == "simple"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
