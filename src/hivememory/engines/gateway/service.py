"""
Gateway Service - 纯数据操作层

职责：
- 执行 L1 拦截器并返回原始结果
- 执行 L2 语义分析器并返回原始结果
- 不处理 fallback、日志、业务逻辑

作者: HiveMemory Team
版本: 2.0
"""

from typing import List, Optional

from hivememory.engines.generation.models import ConversationMessage
from hivememory.engines.gateway.interfaces import BaseInterceptor, BaseSemanticAnalyzer
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    SemanticAnalysisResult,
)

__all__ = ["GatewayService"]


class GatewayService:
    """
    Gateway 数据操作层

    这是 GatewayService 模块的纯数据操作层，负责：
    - 调用 L1 拦截器并返回原始结果
    - 调用 L2 语义分析器并返回原始结果
    - 将结果封装为 GatewayResult

    注意：此类不处理任何业务逻辑，如：
    - 不处理 fallback（由上层 TheEye 处理）
    - 不记录日志（由上层 TheEye 处理）
    - 不添加处理时间（由上层 TheEye 处理）

    Examples:
        >>> from hivememory.engines.gateway.interceptors import RuleInterceptor
        >>> from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer
        >>>
        >>> interceptor = RuleInterceptor()
        >>> analyzer = LLMAnalyzer(llm_service=...)
        >>>
        >>> service = GatewayService(
        ...     interceptor=interceptor,
        ...     semantic_analyzer=analyzer
        ... )
        >>>
        >>> result = service.process("你好，世界！")
    """

    def __init__(
        self,
        interceptor: Optional[BaseInterceptor] = None,
        semantic_analyzer: Optional[BaseSemanticAnalyzer] = None,
    ):
        """
        初始化 GatewayService

        Args:
            interceptor: L1 拦截器实例（可选）
            semantic_analyzer: L2 语义分析器实例（可选）
        """
        self.interceptor = interceptor
        self.semantic_analyzer = semantic_analyzer

    def execute_l1(
        self,
        query: str,
    ) -> Optional[InterceptorResult]:
        """
        执行 L1 拦截器

        Args:
            query: 用户查询字符串

        Returns:
            InterceptorResult 如果拦截成功，None 如果拦截器未配置或未命中
        """
        if self.interceptor is None:
            return None
        return self.interceptor.intercept(query)

    def execute_l2(
        self,
        query: str,
        context: List[ConversationMessage],
    ) -> Optional[SemanticAnalysisResult]:
        """
        执行 L2 语义分析器

        Args:
            query: 用户查询字符串
            context: 对话上下文

        Returns:
            SemanticAnalysisResult 如果分析成功，None 如果分析器未配置
        """
        if self.semantic_analyzer is None:
            return None
        return self.semantic_analyzer.analyze(query, context)

    def process(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
    ) -> GatewayResult:
        """
        执行完整的 Gateway 处理流程

        流程：
        1. 尝试 L1 拦截
        2. 如果 L1 未命中，执行 L2 语义分析
        3. 将 L1/L2 原始结果统一转换为 GatewayResult

        Args:
            query: 用户查询字符串
            context: 对话上下文（可选）

        Returns:
            GatewayResult: GatewayService 对 TheEye 的最终输出

        Note:
            此方法不处理 fallback、不记录日志、不添加处理时间。
            这些业务逻辑由上层 TheEye 处理。
        """
        # L1: 拦截器
        l1_result = self.execute_l1(query)

        if l1_result is not None and l1_result.hit:
            # L1 命中，转换为 GatewayResult
            return GatewayResult(
                intent=l1_result.intent,
                rewritten_query=query,
                search_keywords=[],
                target_filters={},
                worth_saving=False,
                reason=l1_result.reason,
                l1_result=l1_result,
            )

        # L2: 语义分析
        l2_result = self.execute_l2(query, context or [])

        if l2_result is None:
            # L2 禁用，返回保守结果
            return GatewayResult(
                intent=GatewayIntent.CHAT,
                rewritten_query=query,
                search_keywords=[],
                target_filters={},
                worth_saving=False,
                reason="L2 semantic analysis disabled",
                l1_result=l1_result,
            )

        # L2 成功，将 SemanticAnalysisResult 转换为 GatewayResult
        return GatewayResult(
            intent=l2_result.intent,
            rewritten_query=l2_result.rewritten_query,
            search_keywords=l2_result.search_keywords,
            target_filters=l2_result.target_filters,
            worth_saving=l2_result.worth_saving,
            reason=l2_result.reason,
            l1_result=l1_result,
        )


__all__ = [
    "GatewayService"
]