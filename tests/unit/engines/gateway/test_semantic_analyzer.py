
import json
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock

from hivememory.patchouli.config import LLMAnalyzerConfig, LLMConfig
from hivememory.engines.gateway.models import GatewayIntent, SemanticAnalysisResult
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer, GATEWAY_FUNCTION_SCHEMA
from hivememory.utils.json_parser import JSONParseError
from hivememory.engines.gateway.prompts import get_system_prompt


class TestLLMAnalyzer:
    """测试 L2 语义分析器 (乐观检索策略)"""

    @pytest.fixture
    def mock_llm_config(self):
        return LLMConfig(temperature=0.1, max_tokens=100)

    @pytest.fixture
    def mock_llm_service(self, mock_llm_config):
        service = Mock()
        service.model = "mock-model"
        service.config = mock_llm_config
        return service

    def test_init(self, mock_llm_service):
        """测试初始化"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        assert analyzer.llm_service == mock_llm_service
        assert analyzer.config is not None
        assert analyzer.system_prompt is not None

        custom_config = LLMAnalyzerConfig(prompt_variant="simple")
        analyzer_custom = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=custom_config,
            system_prompt="Custom Prompt"
        )
        assert analyzer_custom.config == custom_config
        assert analyzer_custom.system_prompt == "Custom Prompt"

    @pytest.mark.asyncio
    async def test_analyze_flow(self, mock_llm_service):
        """测试正常分析流程 (乐观检索策略)"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=config
        )

        # Mock LLM response (乐观检索策略: 不再需要 intent 和 memory_type)
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()

        arguments = {
            "rewritten_query": "Rewritten Query",
            "search_keywords": ["keyword1"],
            "worth_saving": True,
            "reason": "Test Reason"
        }

        mock_tool_call.function.arguments = json.dumps(arguments)
        mock_message.tool_calls = [mock_tool_call]
        mock_response.choices = [MagicMock(message=mock_message)]

        mock_llm_service.acomplete_with_tools = AsyncMock(return_value=mock_response)

        result = await analyzer.analyze("Query")

        assert isinstance(result, SemanticAnalysisResult)
        assert result.intent == GatewayIntent.RAG  # 乐观策略: 默认 RAG
        assert result.rewritten_query == "Rewritten Query"
        assert result.search_keywords == ["keyword1"]
        # 乐观策略: SemanticAnalysisResult 不再包含 target_filters
        assert result.worth_saving is True
        assert result.reason == "Test Reason"
        assert result.model == "mock-model"

        # Verify LLM call arguments
        mock_llm_service.acomplete_with_tools.assert_called_once()
        call_args = mock_llm_service.acomplete_with_tools.call_args
        assert call_args.kwargs["tools"] == [GATEWAY_FUNCTION_SCHEMA]
        assert call_args.kwargs["messages"] == [
            {"role": "system", "content": analyzer.system_prompt},
            {"role": "user", "content": "Query"},
        ]
        # Verify LLMConfig usage
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_analyze_with_active_topics_menu(self, mock_llm_service):
        """测试带话题菜单的分析"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps({
            "rewritten_query": "Q",
            "search_keywords": [],
            "worth_saving": False,
            "reason": "R"
        })
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_llm_service.acomplete_with_tools = AsyncMock(return_value=mock_response)

        active_topics_menu = '["topic_1: Python学习"]'
        await analyzer.analyze("Query", active_topics_menu=active_topics_menu)

        call_args = mock_llm_service.acomplete_with_tools.call_args
        messages = call_args.kwargs["messages"]
        assert messages == [
            {
                "role": "system",
                "content": get_system_prompt(
                    variant="dispatcher",
                    language=config.prompt_language,
                    active_topics_menu=active_topics_menu,
                ),
            },
            {"role": "user", "content": "Query"},
        ]

    @pytest.mark.asyncio
    async def test_parse_response_invalid_structure(self, mock_llm_service):
        """测试无效响应结构"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        # No choices
        mock_response = MagicMock()
        mock_response.choices = []
        mock_llm_service.acomplete_with_tools = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Invalid response structure"):
            await analyzer.analyze("Query")

    @pytest.mark.asyncio
    async def test_parse_response_no_tool_calls(self, mock_llm_service):
        """测试无 tool calls"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = []
        mock_llm_service.acomplete_with_tools = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="No tool_calls"):
            await analyzer.analyze("Query")

    @pytest.mark.asyncio
    async def test_parse_response_invalid_json(self, mock_llm_service):
        """测试参数非 JSON"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = "invalid-json"
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_llm_service.acomplete_with_tools = AsyncMock(return_value=mock_response)

        with pytest.raises(JSONParseError):
            await analyzer.analyze("Query")
