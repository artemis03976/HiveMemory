"""
HiveMemory - Retrieval 模块接口抽象层

定义了记忆检索模块的核心接口（骨架）。

状态: 待 Stage 2 实现
作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from hivememory.core.models import MemoryAtom, ConversationMessage


class RetrievalRouter(ABC):
    """
    检索路由器接口

    职责:
        判断用户查询是否需要检索记忆。

    实现策略 (计划):
        - 轻量级分类器 (GPT-4o-mini)
        - 关键词匹配 ("昨天"、"之前的"、"项目中的")
    """

    @abstractmethod
    def should_retrieve(
        self,
        query: str,
        context: List[ConversationMessage]
    ) -> bool:
        """
        判断是否需要检索记忆

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            bool: True 表示需要检索
        """
        pass


class QueryProcessor(ABC):
    """
    查询预处理器接口

    职责:
        重写和扩展用户查询，提取过滤条件。

    实现策略 (计划):
        - LLM 驱动的 Query Rewriting
        - 提取时间、类型、Agent 等结构化过滤器
    """

    @abstractmethod
    def process(self, query: str) -> "ProcessedQuery":
        """
        处理查询

        Args:
            query: 原始查询

        Returns:
            ProcessedQuery: 处理后的查询对象
        """
        pass


class MemorySearcher(ABC):
    """
    记忆检索器接口

    职责:
        执行混合检索（向量 + BM25 + 结构化过滤）。

    实现策略 (计划):
        - 向量检索 (Qdrant Dense Vector)
        - 关键词检索 (Qdrant Sparse Vector / BM25)
        - RRF 融合 (Reciprocal Rank Fusion)
        - Reranking (Cross-Encoder)
    """

    @abstractmethod
    def search(
        self,
        query: "ProcessedQuery",
        top_k: int = 5,
        score_threshold: float = 0.75
    ) -> List[MemoryAtom]:
        """
        检索记忆

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            List[MemoryAtom]: 检索结果
        """
        pass


class ProcessedQuery:
    """
    处理后的查询对象 (占位符)

    Attributes:
        semantic_query: 语义查询文本
        keywords: 关键词列表
        filters: 结构化过滤条件
    """
    pass


__all__ = [
    "RetrievalRouter",
    "QueryProcessor",
    "MemorySearcher",
    "ProcessedQuery",
]
