
import json
import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.system.config import LLMAnalyzerConfig, LLMConfig
from hivememory.engines.gateway.models import GatewayIntent, SemanticAnalysisResult
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer
from hivememory.utils.json_parser import JSONParseError
from hivememory.prompts.gateway import get_gateway_system_prompt
from hivememory.i18n import set_default_language


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


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
        assert analyzer.language == "zh"
        assert analyzer.system_prompt is not None
        assert "JSON object" in analyzer.system_prompt

    @pytest.mark.asyncio
    async def test_analyze_flow(self, mock_llm_service):
        """测试正常分析流程 (乐观检索策略)"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=config
        )

        arguments = {
            "rewritten_query": "Rewritten Query",
            "search_keywords": ["keyword1"],
            "worth_saving": True,
            "reason": "Test Reason",
        }

        mock_llm_service.acomplete_json = AsyncMock(return_value=json.dumps(arguments))

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
        mock_llm_service.acomplete_json.assert_called_once()
        call_args = mock_llm_service.acomplete_json.call_args
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

        mock_llm_service.acomplete_json = AsyncMock(return_value=json.dumps({
            "rewritten_query": "Q",
            "search_keywords": [],
            "worth_saving": False,
            "reason": "R"
        }))

        active_topics_menu = '["topic_1: Python学习"]'
        await analyzer.analyze("Query", active_topics_menu=active_topics_menu)

        call_args = mock_llm_service.acomplete_json.call_args
        messages = call_args.kwargs["messages"]
        assert messages == [
            {
                "role": "system",
                "content": get_gateway_system_prompt(
                    language=analyzer.language,
                    active_topics_menu=active_topics_menu,
                ),
            },
            {"role": "user", "content": "Query"},
        ]

    def test_default_language_is_used_for_gateway_prompt(self, mock_llm_service):
        set_default_language("en")
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=config,
        )

        assert analyzer.language == "en"
        assert "You are an OS-level dispatch gateway" in analyzer.system_prompt

    @pytest.mark.asyncio
    async def test_parse_response_missing_required_fields_uses_defaults(self, mock_llm_service):
        """测试缺失字段时使用保守默认值"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        mock_llm_service.acomplete_json = AsyncMock(return_value=json.dumps({}))

        result = await analyzer.analyze("Query")
        assert result.rewritten_query == "Query"
        assert result.search_keywords == []
        assert result.worth_saving is False
        assert result.reason == ""
        assert result.target_topic == "NEW_TOPIC"

    @pytest.mark.asyncio
    async def test_parse_response_invalid_json(self, mock_llm_service):
        """测试 content 非 JSON"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)

        mock_llm_service.acomplete_json = AsyncMock(return_value="no json here")

        with pytest.raises(JSONParseError):
            await analyzer.analyze("Query")
