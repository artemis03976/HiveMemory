"""
HiveMemory - Retrieval 模块接口抽象层

接口列表:
- BaseMemoryRetriever: 记忆检索器接口
- BaseReranker: 重排序器接口
- BaseFusion: 融合算法接口
"""

from abc import ABC, abstractmethod

from hivememory.engines.retrieval.models import RetrievalQuery, SearchResults


# ========== 接口定义 ==========

class BaseMemoryRetriever(ABC):
    """
    记忆检索器接口

    实现策略:
        - 向量检索 (Qdrant Dense Vector)
        - 元数据过滤 (类型、标签、时间、置信度)
        - 结果排序和打分
        - 可选的 RRF 融合和 Reranking
    """

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        score_threshold: float = 0.75
    ) -> SearchResults:
        """
        检索记忆

        Args:
            query: 检索查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            SearchResults: 检索结果集合

        Examples:
            >>> searcher = HybridRetriever(storage)
            >>> results = searcher.retrieve(processed_query, top_k=5)
            >>> for result in results:
            ...     print(result.memory.index.title)
        """
        pass

class BaseFusion(ABC):
    """
    融合算法接口
    
    职责:
        合并多路检索结果。
    """
    
    @abstractmethod
    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults
    ) -> SearchResults:
        """
        融合检索结果
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            
        Returns:
            融合后的结果
        """
        pass


class BaseReranker(ABC):
    """
    重排序器抽象接口

    用于在 RRF 融合后对结果进行精排。
    """

    @abstractmethod
    def rerank(
        self,
        results: SearchResults,
        query: RetrievalQuery
    ) -> SearchResults:
        """
        对检索结果进行重排序

        Args:
            results: RRF 融合后的检索结果
            query: 原始查询

        Returns:
            重排序后的结果
        """
        pass


__all__ = [
    "BaseMemoryRetriever",
    "BaseFusion",
    "BaseReranker",
]
