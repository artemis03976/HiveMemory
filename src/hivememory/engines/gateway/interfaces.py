"""
Global Gateway 接口定义

定义 Gateway 子模块的抽象接口

作者: HiveMemory Team
版本: 2.0
"""

from abc import ABC, abstractmethod

from hivememory.engines.gateway.models import (
    InterceptorResult,
    SemanticAnalysisResult,
)

# ============================================================
# L1: 拦截器接口
# ============================================================


class BaseInterceptor(ABC):
    """
    L1 拦截器抽象基类

    定义快速拦截器的标准接口，用于零开销拦截常见查询。

    实现类:
        - RuleInterceptor: 基于正则规则的拦截器
        - 未来可扩展: MLInterceptor, CacheInterceptor 等

    示例:
        >>> class CustomInterceptor(Interceptor):
        ...     def intercept(
        ...         self,
        ...         query: str,
        ...         *,
        ...         allow_system: bool = True,
        ...     ) -> InterceptorResult | None:
        ...         if query.startswith("!"):
        ...             return InterceptorResult(
        ...                 intent=GatewayIntent.SYSTEM,
        ...                 reason="自定义系统指令",
        ...                 hit=True
        ...             )
        ...         return None
    """

    @abstractmethod
    def intercept(
        self,
        query: str,
        *,
        allow_system: bool = True,
    ) -> InterceptorResult | None:
        """
        执行拦截

        Args:
            query: 用户原始查询

        Returns:
            InterceptorResult if intercepted, None otherwise
                - hit=True: 已拦截，返回结果
                - hit=False 或 None: 未拦截，进入下一阶段
        """
        raise NotImplementedError


# ============================================================
# L2: 语义分析器接口
# ============================================================

class BaseSemanticAnalyzer(ABC):
    """
    L2 语义分析器抽象基类

    定义语义分析的核心接口，负责意图分类、查询重写和元数据提取。

    实现类:
        - LLMAnalyzer: 基于 LLM JSON mode
        - 未来可扩展: HybridAnalyzer, LocalModelAnalyzer 等

    示例:
        >>> analyzer = LLMAnalyzer(llm_service)
        >>> result = await analyzer.analyze(
        ...     query="怎么部署它？",
        ... )
    """

    @abstractmethod
    async def analyze(
        self,
        query: str,
        active_topics_menu: str | None = None,
    ) -> SemanticAnalysisResult:
        """
        执行语义分析

        Args:
            query: 用户原始查询
            active_topics_menu: 活跃话题菜单字符串（用于 Agentic Routing）

        Returns:
            SemanticAnalysisResult: L2 分析器的原始输出，包含意图、重写查询、关键词、路由目标等
        """
        raise NotImplementedError


__all__ = [
    # L1 拦截器
    "BaseInterceptor",
    # L2 语义分析
    "BaseSemanticAnalyzer",
]
