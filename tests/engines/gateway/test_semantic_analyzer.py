
import json
import pytest
from unittest.mock import Mock, MagicMock

from hivememory.core.models import MemoryType
from hivememory.patchouli.config import LLMAnalyzerConfig, LLMConfig
from hivememory.patchouli.protocol.models import QueryFilters
from hivememory.engines.gateway.models import GatewayIntent, SemanticAnalysisResult
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer, GATEWAY_FUNCTION_SCHEMA
from hivememory.core.models import StreamMessage
from hivememory.engines.gateway.prompts import get_system_prompt


class TestLLMAnalyzer:
    """测试 L2 语义分析器"""

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

        custom_config = LLMAnalyzerConfig(context_window=5)
        analyzer_custom = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=custom_config,
            system_prompt="Custom Prompt"
        )
        assert analyzer_custom.config == custom_config
        assert analyzer_custom.system_prompt == "Custom Prompt"

    def test_analyze_flow(self, mock_llm_service):
        """测试正常分析流程"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(
            llm_service=mock_llm_service,
            config=config
        )
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        
        arguments = {
            "intent": "RAG",
            "rewritten_query": "Rewritten Query",
            "search_keywords": ["keyword1"],
            "memory_type": "CODE_SNIPPET",
            "worth_saving": True,
            "reason": "Test Reason"
        }
        
        mock_tool_call.function.arguments = json.dumps(arguments)
        mock_message.tool_calls = [mock_tool_call]
        mock_response.choices = [MagicMock(message=mock_message)]
        
        mock_llm_service.complete_with_tools.return_value = mock_response

        result = analyzer.analyze("Query", [])

        assert isinstance(result, SemanticAnalysisResult)
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "Rewritten Query"
        assert result.search_keywords == ["keyword1"]
        assert result.target_filters.memory_type == MemoryType.CODE_SNIPPET
        assert result.worth_saving is True
        assert result.reason == "Test Reason"
        assert result.model == "mock-model"

        # Verify LLM call arguments
        mock_llm_service.complete_with_tools.assert_called_once()
        call_args = mock_llm_service.complete_with_tools.call_args
        assert call_args.kwargs["tools"] == [GATEWAY_FUNCTION_SCHEMA]
        # Verify LLMConfig usage
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_tokens"] == 100

    def test_analyze_with_context(self, mock_llm_service):
        """测试带上下文的分析"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        
        # Mock response (minimal valid)
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps({
            "intent": "CHAT",
            "rewritten_query": "Q",
            "search_keywords": [],
            "worth_saving": False,
            "reason": "R"
        })
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_llm_service.complete_with_tools.return_value = mock_response

        context = [
            StreamMessage(message_type="user", content="Hi"),
            StreamMessage(message_type="assistant", content="Hello")
        ]
        
        analyzer.analyze("Query", context)
        
        # Check if context was added to messages
        call_args = mock_llm_service.complete_with_tools.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2 # system + user
        last_msg = messages[-1]["content"]
        assert "Hi" in last_msg
        assert "Hello" in last_msg
        assert "Query" in last_msg

    def test_parse_response_invalid_structure(self, mock_llm_service):
        """测试无效响应结构"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        
        # No choices
        mock_response = MagicMock()
        mock_response.choices = []
        mock_llm_service.complete_with_tools.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid response structure"):
            analyzer.analyze("Query", [])

    def test_parse_response_no_tool_calls(self, mock_llm_service):
        """测试无 tool calls"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = []
        mock_llm_service.complete_with_tools.return_value = mock_response
        
        with pytest.raises(ValueError, match="No tool_calls"):
            analyzer.analyze("Query", [])

    def test_parse_response_invalid_json(self, mock_llm_service):
        """测试参数非 JSON"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = "invalid-json"
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_llm_service.complete_with_tools.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            analyzer.analyze("Query", [])
            
    def test_build_filters(self, mock_llm_service):
        """测试过滤器构建"""
        config = LLMAnalyzerConfig()
        analyzer = LLMAnalyzer(llm_service=mock_llm_service, config=config)
        
        # Private method test
        filters = analyzer._build_filters("CODE_SNIPPET")
        assert isinstance(filters, QueryFilters)
        assert filters.memory_type == MemoryType.CODE_SNIPPET
        
        filters = analyzer._build_filters(None)
        assert isinstance(filters, QueryFilters)
        assert filters.is_empty()
        
        # Disable filter config
        analyzer.config.enable_memory_type_filter = False
        filters = analyzer._build_filters("CODE_SNIPPET")
        assert isinstance(filters, QueryFilters)
        assert filters.is_empty()
