"""
Global Gateway 接口定义

定义 Gateway 子模块的抽象接口

作者: HiveMemory Team
版本: 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from hivememory.generation.models import ConversationMessage
from hivememory.gateway.models import (
    ContentPayload,
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    MemorySignal,
)


# ============================================================
# L1: 拦截器接口
# ============================================================


class Interceptor(ABC):
    """
    L1 拦截器抽象基类

    定义快速拦截器的标准接口，用于零开销拦截常见查询。

    实现类:
        - RuleInterceptor: 基于正则规则的拦截器
        - 未来可扩展: MLInterceptor, CacheInterceptor 等

    示例:
        >>> class CustomInterceptor(Interceptor):
        ...     def intercept(self, query: str) -> Optional[InterceptorResult]:
        ...         if query.startswith("!"):
        ...             return InterceptorResult(
        ...                 intent=GatewayIntent.SYSTEM,
        ...                 reason="自定义系统指令",
        ...                 hit=True
        ...             )
        ...         return None
    """

    @abstractmethod
    def intercept(self, query: str) -> Optional[InterceptorResult]:
        """
        执行拦截

        Args:
            query: 用户原始查询

        Returns:
            InterceptorResult if intercepted, None otherwise
                - hit=True: 已拦截，返回结果
                - hit=False 或 None: 未拦截，进入下一阶段
        """
        pass


# ============================================================
# L2: 语义分析器接口
# ============================================================


class SemanticAnalyzer(ABC):
    """
    L2 语义分析器抽象基类

    定义语义分析的核心接口，负责意图分类、查询重写和元数据提取。

    实现类:
        - LLMAnalyzer: 基于 LLM + Function Calling
        - 未来可扩展: HybridAnalyzer, LocalModelAnalyzer 等

    示例:
        >>> analyzer = LLMAnalyzer(llm_service)
        >>> result = analyzer.analyze(
        ...     query="怎么部署它？",
        ...     context=[ConversationMessage(role="user", content="贪吃蛇游戏")]
        ... )
    """

    @abstractmethod
    def analyze(
        self,
        query: str,
        context: List[ConversationMessage],
    ) -> GatewayResult:
        """
        执行语义分析

        Args:
            query: 用户原始查询
            context: 对话上下文（用于指代消解）

        Returns:
            GatewayResult: 包含意图、重写查询、关键词、记忆信号的结构化结果
        """
        pass


# ============================================================
# Gateway 主接口
# ============================================================


class Gateway(ABC):
    """
    Gateway 接口

    示例:
        >>> def process_with_gateway(gateway: Gateway, query: str):
        ...     result = gateway.process(query)
        ...     return result
    """

    def process(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
    ) -> GatewayResult:
        """
        处理用户查询

        Args:
            query: 用户原始查询
            context: 对话上下文（可选）

        Returns:
            GatewayResult: 结构化分析结果
        """
        pass


__all__ = [
    # L1 拦截器
    "Interceptor",
    # L2 语义分析
    "SemanticAnalyzer",
    # Gateway 主接口
    "Gateway",
]
