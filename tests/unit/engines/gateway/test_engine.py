"""
GatewayEngine 单元测试

测试覆盖:
- L1 拦截命中 / 未命中分支
- L2 语义分析 fallback
- GatewayResult 字段映射
- 参数传递 (active_topics_menu)
"""

import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    SemanticAnalysisResult,
)
from hivememory.system.gateway.commands import CommandParseResult, CommandParseStatus


class TestGatewayEngine:
    """GatewayEngine 编排逻辑单元测试"""

    def setup_method(self):
        self.mock_interceptor = Mock()
        self.mock_analyzer = Mock()
        self.mock_analyzer.analyze = AsyncMock()
        self.engine = GatewayEngine(
            interceptor=self.mock_interceptor,
            semantic_analyzer=self.mock_analyzer,
        )

    # ========== L1 拦截命中 ==========

    @pytest.mark.asyncio
    async def test_l1_hit_returns_early(self):
        """L1 命中时直接返回，不调用 L2"""
        self.mock_interceptor.intercept.return_value = InterceptorResult(
            intent=GatewayIntent.CHAT, reason="闲聊", hit=True
        )

        result = await self.engine.process("你好")

        self.mock_interceptor.intercept.assert_called_once_with("你好")
        self.mock_analyzer.analyze.assert_not_called()
        assert result.intent == GatewayIntent.CHAT

    @pytest.mark.asyncio
    async def test_l1_hit_sets_worth_saving_false(self):
        """L1 命中时 worth_saving=False"""
        self.mock_interceptor.intercept.return_value = InterceptorResult(
            intent=GatewayIntent.SYSTEM, reason="系统指令", hit=True
        )

        result = await self.engine.process("/help")

        assert result.worth_saving is False

    @pytest.mark.asyncio
    async def test_l1_hit_sets_target_topic_new(self):
        """L1 命中时 target_topic='NEW_TOPIC'"""
        self.mock_interceptor.intercept.return_value = InterceptorResult(
            intent=GatewayIntent.CHAT, reason="闲聊", hit=True
        )

        result = await self.engine.process("你好")

        assert result.target_topic == "NEW_TOPIC"

    @pytest.mark.asyncio
    async def test_l1_hit_preserves_l1_result(self):
        """L1 命中时 l1_result 保留在输出中"""
        l1 = InterceptorResult(intent=GatewayIntent.CHAT, reason="闲聊", hit=True)
        self.mock_interceptor.intercept.return_value = l1

        result = await self.engine.process("你好")

        assert result.l1_result is l1

    @pytest.mark.asyncio
    async def test_l1_hit_preserves_command_result(self):
        command = CommandParseResult(
            command_id="system.help",
            raw_input="/help",
            name="/help",
            tokens=["/help"],
            matched_alias="/help",
            parse_status=CommandParseStatus.MATCHED,
        )
        self.mock_interceptor.intercept.return_value = InterceptorResult(
            intent=GatewayIntent.SYSTEM,
            reason="system command",
            hit=True,
            command=command,
        )

        result = await self.engine.process("/help")

        assert result.command is command
        self.mock_analyzer.analyze.assert_not_called()

    # ========== L1 未命中 → L2 ==========

    @pytest.mark.asyncio
    async def test_l1_miss_falls_through_to_l2(self):
        """L1 未命中时走 L2 语义分析"""
        self.mock_interceptor.intercept.return_value = InterceptorResult(
            intent=GatewayIntent.RAG, reason="", hit=False
        )
        self.mock_analyzer.analyze.return_value = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="Python 快排实现",
            search_keywords=["快排", "Python"],
            worth_saving=True,
            reason="知识查询",
            target_topic="topic_001",
        )

        result = await self.engine.process("帮我写个快排")

        self.mock_analyzer.analyze.assert_called_once()
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "Python 快排实现"

    @pytest.mark.asyncio
    async def test_l1_none_falls_through_to_l2(self):
        """L1 返回 None 时走 L2"""
        self.mock_interceptor.intercept.return_value = None
        self.mock_analyzer.analyze.return_value = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="query",
            search_keywords=[],
            worth_saving=False,
            reason="",
        )

        result = await self.engine.process("test")

        self.mock_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_l2_result_fields_mapped_correctly(self):
        """L2 结果字段正确映射到 GatewayResult"""
        self.mock_interceptor.intercept.return_value = None
        l2 = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="重写后的查询",
            search_keywords=["kw1", "kw2"],
            worth_saving=True,
            reason="有价值的查询",
            target_topic="topic_abc",
        )
        self.mock_analyzer.analyze.return_value = l2

        result = await self.engine.process("原始查询")

        assert result.intent == l2.intent
        assert result.rewritten_query == l2.rewritten_query
        assert result.search_keywords == l2.search_keywords
        assert result.worth_saving == l2.worth_saving
        assert result.reason == l2.reason
        assert result.target_topic == l2.target_topic

    @pytest.mark.asyncio
    async def test_l1_result_preserved_when_l2_used(self):
        """L1 未命中时 l1_result 仍保留在最终输出中"""
        l1 = InterceptorResult(intent=GatewayIntent.RAG, reason="", hit=False)
        self.mock_interceptor.intercept.return_value = l1
        self.mock_analyzer.analyze.return_value = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="q",
            search_keywords=[],
            worth_saving=False,
            reason="",
        )

        result = await self.engine.process("test")

        assert result.l1_result is l1
        assert result.command is None

    # ========== 参数传递 ==========

    @pytest.mark.asyncio
    async def test_process_with_active_topics_menu(self):
        """带话题菜单调用时正确传递给 L2"""
        self.mock_interceptor.intercept.return_value = None
        self.mock_analyzer.analyze.return_value = SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query="q",
            search_keywords=[],
            worth_saving=False,
            reason="",
        )
        menu = '["topic_1: Python学习"]'

        await self.engine.process("test", active_topics_menu=menu)

        call_args = self.mock_analyzer.analyze.call_args
        assert call_args[0][0] == "test"
        assert call_args[1]["active_topics_menu"] == menu
