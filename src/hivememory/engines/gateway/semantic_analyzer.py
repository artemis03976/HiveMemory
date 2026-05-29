"""
L2 语义分析器实现

提供基于 LLM JSON mode 的语义分析实现。

作者: HiveMemory Team
版本: 3.0 (Phase 4.5 Agentic Dispatcher)
"""

import logging
from typing import Optional

from hivememory.system.config import LLMAnalyzerConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.engines.gateway.interfaces import BaseSemanticAnalyzer
from hivememory.engines.gateway.models import (
    GatewayIntent,
    SemanticAnalysisResult,
)
from hivememory.i18n import resolve_language
from hivememory.prompts.gateway import get_gateway_system_prompt
from hivememory.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


class LLMAnalyzer(BaseSemanticAnalyzer):
    """
    基于 LLM JSON mode 的语义分析器
    """

    def __init__(
        self,
        config: LLMAnalyzerConfig,
        llm_service: BaseLLMService,
        system_prompt: Optional[str] = None,
        default_language: str | None = None,
    ):
        """
        初始化 LLMAnalyzer

        Args:
            config: LLMAnalyzerConfig 配置对象
            llm_service: LLM 服务实例
            system_prompt: 自定义系统提示词（可选）
            default_language: 全局默认语言（可选）
        """
        self.config = config
        self.llm_service = llm_service
        self.language = resolve_language(
            component_language=self.config.prompt_language,
            default_language=default_language,
        ).value
        self.system_prompt = system_prompt or get_gateway_system_prompt(
            variant=self.config.prompt_variant,
            language=self.language,
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
                language=self.language,
                active_topics_menu=active_topics_menu,
            )
        else:
            system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # 调用 LLM (使用 JSON mode, service 内部负责不支持时降级)
        try:
            content = await self.llm_service.acomplete_json(
                messages=messages,
                temperature=self.llm_service.config.temperature,
                max_tokens=self.llm_service.config.max_tokens,
            )

            return self._parse_json_response(content, query)
            
        except Exception as e:
            logger.error(f"LLM 语义分析失败: {e}", exc_info=True)
            raise e

    def _parse_json_response(
        self,
        content: str,
        original_query: str,
    ) -> SemanticAnalysisResult:
        arguments = parse_llm_json(content)

        return SemanticAnalysisResult(
            intent=GatewayIntent.RAG,
            rewritten_query=arguments.get("rewritten_query") or original_query,
            search_keywords=arguments.get("search_keywords", []),
            worth_saving=arguments.get("worth_saving", False),
            reason=arguments.get("reason", ""),
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
    default_language: str | None = None,
) -> BaseSemanticAnalyzer:
    """
    创建 L2 语义分析器实例

    Args:
        config: L2 分析器配置
        llm_service: LLM 服务实例
        default_language: 全局默认语言（可选）

    Returns:
        BaseSemanticAnalyzer: LLMAnalyzer 或 NoOpSemanticAnalyzer
    """
    if config.enabled:
        logger.info("Gateway L2 语义分析器已启用")
        return LLMAnalyzer(
            config,
            llm_service,
            default_language=default_language,
        )
    else:
        logger.info("Gateway L2 语义分析器已禁用 (No-Op)")
        return NoOpSemanticAnalyzer()


__all__ = [
    "LLMAnalyzer",
    "NoOpSemanticAnalyzer",
    "create_semantic_analyzer",
]
