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
        - 可选的 RRF 融合
    """

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
    ) -> SearchResults:
        """
        检索记忆

        分数阈值不由调用方传入：只有 Dense 召回按自身配置的 cosine 阈值粗筛。

        Args:
            query: 检索查询
            top_k: 返回数量；None 使用实现的配置默认值

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
        合并多路检索结果，按融合分数排序后全部返回，不截取；
        条数由检索器按调用方请求的候选数截取，最终条数在重排之后确定。
    """

    @abstractmethod
    def fuse(self, dense_results: SearchResults, sparse_results: SearchResults) -> SearchResults:
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

    RetrievalEngine 中召回之后的独立阶段，对任意检索器的召回结果进行精排。
    """

    def candidate_count(self, top_k: int) -> int:
        """重排需要向检索器索取的候选数；默认与最终返回数相同。"""
        return top_k

    @abstractmethod
    async def rerank(self, results: SearchResults, query: RetrievalQuery) -> SearchResults:
        """
        对检索结果进行重排序

        Args:
            results: 检索器返回的召回结果
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
