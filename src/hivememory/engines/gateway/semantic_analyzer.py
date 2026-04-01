"""
L2 语义分析器实现

提供基于 LLM + Function Calling 的语义分析实现。

作者: HiveMemory Team
版本: 3.0 (Phase 4.5 Agentic Dispatcher)
"""

import logging
from typing import Any, Optional

from hivememory.patchouli.config import LLMAnalyzerConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.engines.gateway.interfaces import BaseSemanticAnalyzer
from hivememory.engines.gateway.models import (
    GatewayIntent,
    SemanticAnalysisResult,
)
from hivememory.prompts.gateway import get_gateway_system_prompt
from hivememory.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


# Function Calling Schema 定义
GATEWAY_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_user_query",
        "description": "分析用户查询，路由到目标话题，重写查询并评估记忆价值",
        "parameters": {
            "type": "object",
            "properties": {
                "target_topic": {
                    "type": "string",
                    "description": "匹配的活跃话题 ID，或 'NEW_TOPIC' 表示新话题",
                },
                "rewritten_query": {
                    "type": "string",
                    "description": "指代消解后的完整、独立的查询",
                },
                "search_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用于稀疏检索的关键词（3-5个）",
                },
                "worth_saving": {
                    "type": "boolean",
                    "description": "是否值得保存为长期记忆",
                },
                "reason": {
                    "type": "string",
                    "description": "判断理由",
                },
                "new_topic_title": {
                    "type": "string",
                    "description": "新话题的简短标题（仅当 target_topic 为 NEW_TOPIC 时填写）",
                },
                "new_topic_summary": {
                    "type": "string",
                    "description": "新话题的一句话摘要（仅当 target_topic 为 NEW_TOPIC 时填写）",
                },
            },
            "required": [
                "target_topic",
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
        self.system_prompt = system_prompt or get_gateway_system_prompt(
            variant=self.config.prompt_variant,
            language=self.config.prompt_language,
        )

    async def analyze(
        self,
        query: str,
        active_topics_menu: Optional[str] = None,
    ) -> SemanticAnalysisResult:
        # 构建系统提示词：有话题菜单时使用 dispatcher 模式
        if active_topics_menu:
            system_prompt = get_gateway_system_prompt(
                variant="dispatcher",
                language=self.config.prompt_language,
                active_topics_menu=active_topics_menu,
            )
        else:
            system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # 调用 LLM (使用 Function Calling)
        try:
            response = await self.llm_service.acomplete_with_tools(
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

        arguments = parse_llm_json(tool_call.function.arguments)

        # 构建 SemanticAnalysisResult (乐观检索策略)
        return SemanticAnalysisResult(
            intent=GatewayIntent.RAG,  # 乐观策略：默认所有查询都可能需要检索
            rewritten_query=arguments["rewritten_query"],
            search_keywords=arguments.get("search_keywords", []),
            worth_saving=arguments["worth_saving"],
            reason=arguments["reason"],
            target_topic=arguments.get("target_topic", "NEW_TOPIC"),
            new_topic_title=arguments.get("new_topic_title"),
            new_topic_summary=arguments.get("new_topic_summary"),
            model=self.llm_service.model,
        )


class NoOpSemanticAnalyzer(BaseSemanticAnalyzer):
    """
    No-Op 语义分析器

    不执行任何分析操作，返回默认的保守结果。
    用于在配置未启用 L2 分析时作为默认实现。
    """

    async def analyze(
        self,
        query: str,
        active_topics_menu: Optional[str] = None,
    ) -> SemanticAnalysisResult:
        """
        执行语义分析 (No-Op)

        Args:
            query: 用户原始查询
            active_topics_menu: 活跃话题菜单（忽略）

        Returns:
            SemanticAnalysisResult: 默认结果
        """
        return SemanticAnalysisResult(
            intent=GatewayIntent.RAG,  # 乐观策略：即使禁用也默认 RAG
            rewritten_query=query,
            search_keywords=[],
            worth_saving=False,
            reason="L2 semantic analysis disabled",
            target_topic="NEW_TOPIC",
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
