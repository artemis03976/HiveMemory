"""
Gateway 单元测试

测试 Gateway 的各个组件：
- Gateway 数据模型
- GatewayResult 回退机制
- GatewayService

作者: HiveMemory Team
版本: 2.2 (乐观检索策略)
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from hivememory.patchouli.config import (
    MemoryGatewayConfig,
    RuleInterceptorConfig,
    LLMAnalyzerConfig,
)
from hivememory.core.models import MemoryType
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    SemanticAnalysisResult,
)
from hivememory.engines.gateway.interceptors import RuleInterceptor, NoOpInterceptor
from hivememory.engines.gateway.semantic_analyzer import NoOpSemanticAnalyzer
from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.prompts.gateway import get_gateway_system_prompt


class TestGatewayModels:
    """测试 Gateway 数据模型 (乐观检索策略)"""

    def test_gateway_intent_enum(self):
        """测试意图枚举 (乐观策略下只保留 RAG/CHAT/SYSTEM)"""
        assert GatewayIntent.RAG.value == "RAG"
        assert GatewayIntent.CHAT.value == "CHAT"
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
        """测试 GatewayResult 默认值 (乐观策略: 无 target_filters)"""
        result = GatewayResult(
            rewritten_query="测试查询",
            worth_saving=False,
            reason="测试"
        )
        assert result.rewritten_query == "测试查询"
        assert result.search_keywords == []
        assert result.intent == GatewayIntent.RAG  # 乐观策略默认 RAG
        assert result.processing_time_ms == 0.0

    def test_gateway_result_fallback(self):
        """测试 Gateway 结果回退 (乐观策略: 回退也是 RAG)"""
        result = GatewayResult.fallback("原始查询")
        assert result.intent == GatewayIntent.RAG  # 乐观策略
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
        """测试 L2 语义分析结果模型 (乐观策略: 无 target_filters)"""
        result = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="如何部署 Python 项目",
            search_keywords=["Python", "部署"],
            worth_saving=True,
            reason="技术问题具有长期参考价值",
            model="gpt-4o-mini"
        )
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "如何部署 Python 项目"
        assert len(result.search_keywords) == 2
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
            intent=GatewayIntent.RAG,
            rewritten_query="你好",
            worth_saving=False,
            reason="闲聊"
        )
        assert result_no_l1.is_l1_intercepted is False


class TestGatewayEngine:
    """测试 GatewayEngine (乐观检索策略)"""

    def test_init_with_interceptor(self):
        """测试使用拦截器初始化"""
        config = RuleInterceptorConfig()
        interceptor = RuleInterceptor(config=config)
        engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=NoOpSemanticAnalyzer()
        )

        assert engine.interceptor is not None
        assert isinstance(engine.semantic_analyzer, NoOpSemanticAnalyzer)

    @pytest.mark.asyncio
    async def test_process_l1_hit_chat(self):
        """测试 L1 命中路径 (CHAT 拦截直接返回 CHAT 意图)"""
        config = RuleInterceptorConfig()
        interceptor = RuleInterceptor(config=config)
        engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=NoOpSemanticAnalyzer()
        )

        result = await engine.process("你好")

        assert result.intent == GatewayIntent.CHAT
        assert result.rewritten_query == "你好"
        assert result.is_l1_intercepted is True

    @pytest.mark.asyncio
    async def test_process_l1_hit_system(self):
        """测试 L1 命中 SYSTEM 指令 (保持 SYSTEM 意图)"""
        config = RuleInterceptorConfig()
        interceptor = RuleInterceptor(config=config)
        engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=NoOpSemanticAnalyzer()
        )

        result = await engine.process("/clear")

        # SYSTEM 指令保持原意图
        assert result.intent == GatewayIntent.SYSTEM
        assert result.is_l1_intercepted is True

    @pytest.mark.asyncio
    async def test_process_l1_no_hit_no_l2(self):
        """测试 L1 未命中且 L2 禁用 (乐观策略: 默认 RAG)"""
        engine = GatewayEngine(
            interceptor=NoOpInterceptor(),
            semantic_analyzer=NoOpSemanticAnalyzer()
        )

        result = await engine.process("如何部署项目？")

        # 乐观策略：NoOpSemanticAnalyzer 也返回 RAG
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "如何部署项目？"
        assert result.search_keywords == []

    @pytest.mark.asyncio
    async def test_process_with_mock_l2(self):
        """测试带 Mock L2 的完整流程"""
        config = RuleInterceptorConfig()
        interceptor = RuleInterceptor(config=config)

        # Mock L2 分析器 (乐观策略: 无 target_filters)
        mock_analyzer = Mock()
        mock_analyzer.analyze = AsyncMock(return_value=SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="如何部署 Python 项目",
            search_keywords=["Python", "部署"],
            worth_saving=True,
            reason="技术问题",
            model="gpt-4o-mini"
        ))

        engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=mock_analyzer
        )

        # L1 不会拦截这个查询，会走 L2
        result = await engine.process("怎么部署它？")

        assert result.intent == GatewayIntent.RAG
        assert "Python" in result.rewritten_query
        assert result.search_keywords == ["Python", "部署"]
        assert result.worth_saving is True
        assert result.is_l1_intercepted is False


class TestMemoryGatewayConfig:
    """测试 Gateway 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = MemoryGatewayConfig()
        assert config.interceptor.enabled is True
        assert config.analyzer.enabled is True

        # 验证子配置
        assert isinstance(config.interceptor, RuleInterceptorConfig)
        assert config.interceptor.enable_system is True
        assert config.interceptor.enable_chat is True

        assert isinstance(config.analyzer, LLMAnalyzerConfig)
        assert config.analyzer.prompt_variant == "default"

    def test_custom_config(self):
        """测试自定义配置"""
        config = MemoryGatewayConfig(
            interceptor=RuleInterceptorConfig(enabled=False),
            analyzer=LLMAnalyzerConfig(
                prompt_variant="simple"
            )
        )
        assert config.interceptor.enabled is False
        assert config.analyzer.prompt_variant == "simple"


class TestSystemPrompts:
    """测试 System Prompt (乐观检索策略)"""

    def test_get_gateway_system_prompt(self):
        # 默认使用中文
        prompt = get_gateway_system_prompt()
        assert "OS 级别的调度网关" in prompt
        assert "无" in prompt

        # 测试 variant 已被忽略，行为一致
        prompt = get_gateway_system_prompt(variant="simple")
        assert prompt == get_gateway_system_prompt()

        # 测试英文
        prompt = get_gateway_system_prompt(language="en")
        assert "OS-level dispatch gateway" in prompt
        assert "None" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
