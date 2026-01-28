"""
L2 语义分析器实现

提供基于 LLM + Function Calling 的语义分析实现。

作者: HiveMemory Team
版本: 2.0
"""

import json
import logging
from typing import Any, List, Optional

from hivememory.patchouli.config import LLMAnalyzerConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.engines.gateway.interfaces import BaseSemanticAnalyzer
from hivememory.core.models import StreamMessage
from hivememory.engines.gateway.models import (
    GatewayIntent,
    SemanticAnalysisResult,
)
from hivememory.engines.gateway.prompts import get_system_prompt
from hivememory.patchouli.protocol.models import QueryFilters
from hivememory.core.models import MemoryType

logger = logging.getLogger(__name__)


# Function Calling Schema 定义
GATEWAY_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_user_query",
        "description": "分析用户查询的意图、重写查询并评估记忆价值",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["RAG", "CHAT", "TOOL", "SYSTEM"],
                    "description": "用户意图分类",
                },
                "rewritten_query": {
                    "type": "string",
                    "description": "指代消解后的完整、独立的查询",
                },
                "search_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用于稀疏检索的关键词",
                },
                "memory_type": {
                    "type": "string",
                    "enum": [
                        "CODE_SNIPPET",
                        "FACT",
                        "URL_RESOURCE",
                        "REFLECTION",
                        "USER_PROFILE",
                        None,
                    ],
                    "description": "记忆类型过滤（可选）",
                },
                "worth_saving": {
                    "type": "boolean",
                    "description": "是否值得保存为长期记忆",
                },
                "reason": {
                    "type": "string",
                    "description": "判断理由",
                },
            },
            "required": [
                "intent",
                "rewritten_query",
                "search_keywords",
                "worth_saving",
                "reason",
            ],
        },
    },
}


class LLMAnalyzer(BaseSemanticAnalyzer):
    """
    基于 LLM + Function Calling 的语义分析器
    """

    def __init__(
        self,
        config: LLMAnalyzerConfig,
        llm_service: BaseLLMService,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 LLMAnalyzer

        Args:
            config: LLMAnalyzerConfig 配置对象
            llm_service: LLM 服务实例
            system_prompt: 自定义系统提示词（可选）
        """
        self.config = config
        self.llm_service = llm_service
        self.system_prompt = system_prompt or get_system_prompt(
            variant=self.config.prompt_variant,
            language=self.config.prompt_language,
        )

    def analyze(
        self,
        query: str,
        context: List[StreamMessage],
    ) -> SemanticAnalysisResult:
        """
        执行语义分析

        使用 Function Calling 调用 LLM，获取结构化输出。

        Args:
            query: 用户原始查询
            context: 对话上下文（用于指代消解）

        Returns:
            SemanticAnalysisResult: L2 分析器的原始输出

        Raises:
            Exception: LLM 调用失败时抛出异常，由 GatewayEngine 处理回退
        """
        # 构建消息
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加上下文（最近 N 条）
        if context and self.config.context_window > 0:
            context_str = self._format_context(context[-self.config.context_window :])
            messages.append(
                {
                    "role": "user",
                    "content": f"最近对话:\n{context_str}\n\n当前查询: {query}",
                }
            )
        else:
            messages.append({"role": "user", "content": query})

        # 调用 LLM (使用 Function Calling)
        try:
            response = self.llm_service.complete_with_tools(
                messages=messages,
                tools=[GATEWAY_FUNCTION_SCHEMA],
                tool_choice={
                    "type": "function",
                    "function": {"name": "analyze_user_query"},
                },
                temperature=self.llm_service.config.temperature,
                max_tokens=self.llm_service.config.max_tokens,
            )

            # 解析 Function Call 结果
            return self._parse_function_call_response(response, query)
            
        except Exception as e:
            logger.error(f"LLM 语义分析失败: {e}", exc_info=True)
            raise e

    def _format_context(self, messages: List[StreamMessage]) -> str:
        """
        格式化上下文为字符串

        Args:
            messages: 对话消息列表

        Returns:
            str: 格式化后的上下文字符串
        """
        lines = []
        for msg in messages:
            role_name = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)

    def _parse_function_call_response(
        self,
        response: Any,
        original_query: str,
    ) -> SemanticAnalysisResult:
        """
        解析 LLM Function Calling 响应

        Args:
            response: litellm 返回的响应对象
            original_query: 原始查询（用于回退）

        Returns:
            SemanticAnalysisResult

        Raises:
            ValueError: 响应结构无效时抛出
            json.JSONDecodeError: JSON 解析失败时抛出
        """
        # 检查响应结构
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("Invalid response structure: no choices")

        message = response.choices[0].message

        # 检查是否有 tool_calls
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            raise ValueError("No tool_calls in response")

        tool_call = message.tool_calls[0]

        # 解析 function arguments
        if not hasattr(tool_call, "function") or not hasattr(
            tool_call.function, "arguments"
        ):
            raise ValueError("Invalid tool_call structure")

        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        # 构建 SemanticAnalysisResult
        return SemanticAnalysisResult(
            intent=GatewayIntent(arguments["intent"]),
            rewritten_query=arguments["rewritten_query"],
            search_keywords=arguments.get("search_keywords", []),
            target_filters=self._build_filters(arguments.get("memory_type")),
            worth_saving=arguments["worth_saving"],
            reason=arguments["reason"],
            model=self.llm_service.model,
        )

    def _build_filters(self, memory_type: Optional[str]) -> QueryFilters:
        """
        构建过滤条件

        Args:
            memory_type: 记忆类型（可选）

        Returns:
            QueryFilters: 结构化过滤条件
        """
        if not memory_type or not self.config.enable_memory_type_filter:
            return QueryFilters()

        # 将字符串转换为 MemoryType 枚举
        try:
            mt = MemoryType(memory_type)
            return QueryFilters(memory_type=mt)
        except (ValueError, TypeError):
            logger.warning(f"无效的 memory_type: {memory_type}, 返回空过滤条件")
            return QueryFilters()


class NoOpSemanticAnalyzer(BaseSemanticAnalyzer):
    """
    No-Op 语义分析器

    不执行任何分析操作，返回默认的保守结果。
    用于在配置未启用 L2 分析时作为默认实现。
    """

    def analyze(
        self,
        query: str,
        context: List[StreamMessage],
    ) -> SemanticAnalysisResult:
        """
        执行语义分析 (No-Op)

        Args:
            query: 用户原始查询
            context: 对话上下文

        Returns:
            SemanticAnalysisResult: 默认结果
                - intent: CHAT
                - rewritten_query: original query
                - worth_saving: False
                - reason: "L2 semantic analysis disabled"
        """
        return SemanticAnalysisResult(
            intent=GatewayIntent.CHAT,
            rewritten_query=query,
            search_keywords=[],
            target_filters=QueryFilters(),
            worth_saving=False,
            reason="L2 semantic analysis disabled",
            model=None,
        )


def create_semantic_analyzer(
    config: LLMAnalyzerConfig,
    llm_service: BaseLLMService,
) -> BaseSemanticAnalyzer:
    """
    创建 L2 语义分析器实例

    Args:
        config: L2 分析器配置
        llm_service: LLM 服务实例

    Returns:
        BaseSemanticAnalyzer: LLMAnalyzer 或 NoOpSemanticAnalyzer
    """
    if config.enabled:
        logger.info("Gateway L2 语义分析器已启用")
        return LLMAnalyzer(config, llm_service)
    else:
        logger.info("Gateway L2 语义分析器已禁用 (No-Op)")
        return NoOpSemanticAnalyzer()


__all__ = [
    "LLMAnalyzer",
    "NoOpSemanticAnalyzer",
    "create_semantic_analyzer",
]
