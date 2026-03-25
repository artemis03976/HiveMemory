"""
Gateway Engine - 纯数据操作层

职责：
- 执行 L1 拦截器并返回原始结果
- 执行 L2 语义分析器并返回原始结果
- 不处理 fallback、日志、业务逻辑

作者: HiveMemory Team
版本: 2.1 (乐观检索策略)
"""

from typing import Optional
from hivememory.engines.gateway.interfaces import BaseInterceptor, BaseSemanticAnalyzer
from hivememory.engines.gateway.models import GatewayResult, GatewayIntent


class GatewayEngine:
    """
    Gateway 数据操作层

    这是 GatewayEngine 模块的纯数据操作层，负责：
    - 调用 L1 拦截器并返回原始结果
    - 调用 L2 语义分析器并返回原始结果
    - 将结果封装为 GatewayResult

    乐观检索策略：
    - 不再生成 target_filters，过滤条件由 RetrievalFamiliar 动态创建
    - 默认意图为 RAG，让检索层决定是否有相关记忆

    Examples:
        >>> from hivememory.engines.gateway.interceptors import RuleInterceptor
        >>> from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer
        >>>
        >>> interceptor = RuleInterceptor()
        >>> analyzer = LLMAnalyzer(llm_service=...)
        >>>
        >>> engine = GatewayEngine(
        ...     interceptor=interceptor,
        ...     semantic_analyzer=analyzer
        ... )
        >>>
        >>> result = await engine.process("你好，世界！")
    """

    def __init__(
        self,
        interceptor: BaseInterceptor,
        semantic_analyzer: BaseSemanticAnalyzer,
    ):
        """
        初始化 GatewayEngine

        Args:
            interceptor: L1 拦截器实例
            semantic_analyzer: L2 语义分析器实例
        """
        self.interceptor = interceptor
        self.semantic_analyzer = semantic_analyzer

    async def process(
        self,
        query: str,
        active_topics_menu: Optional[str] = None,
    ) -> GatewayResult:
        """
        执行完整的 Gateway 处理流程

        流程：
        1. 尝试 L1 拦截
        2. 如果 L1 未命中，执行 L2 语义分析（含话题路由）
        3. 将 L1/L2 原始结果统一转换为 GatewayResult

        Args:
            query: 用户查询字符串
            active_topics_menu: 活跃话题菜单字符串（用于 Agentic Routing）

        Returns:
            GatewayResult: GatewayEngine 对 TheEye 的最终输出

        Note:
            此方法不处理 fallback、不记录日志、不添加处理时间。
            这些业务逻辑由上层 TheEye 处理。
        """
        # L1: 拦截器
        l1_result = self.interceptor.intercept(query)

        if l1_result is not None and l1_result.hit:
            return GatewayResult(
                intent=l1_result.intent,
                rewritten_query=query,
                search_keywords=[],
                worth_saving=False,
                reason=l1_result.reason,
                l1_result=l1_result,
                target_topic="NEW_TOPIC",  # L1 拦截不做路由
            )

        # L2: 语义分析（含话题路由）
        l2_result = await self.semantic_analyzer.analyze(
            query, active_topics_menu=active_topics_menu
        )

        # L2 成功，将 SemanticAnalysisResult 转换为 GatewayResult
        return GatewayResult(
            intent=l2_result.intent,
            rewritten_query=l2_result.rewritten_query,
            search_keywords=l2_result.search_keywords,
            worth_saving=l2_result.worth_saving,
            reason=l2_result.reason,
            l1_result=l1_result,
            target_topic=l2_result.target_topic,
            new_topic_title=l2_result.new_topic_title,
            new_topic_summary=l2_result.new_topic_summary,
        )


__all__ = [
    "GatewayEngine"
]
