"""
HiveMemory - Retrieval 模块接口抽象层

定义了记忆检索模块的所有核心接口，遵循依赖倒置原则，便于扩展和测试。

接口列表:
- BaseMemoryRetriever: 记忆检索器接口
- BaseContextRenderer: 上下文渲染器接口
- BaseReranker: 重排序器接口

作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from hivememory.engines.retrieval.models import RetrievalQuery, SearchResults, RenderFormat


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
    def retrieve(
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


class BaseContextRenderer(ABC):
    """
    上下文渲染器接口

    职责:
        将检索到的记忆渲染为适合注入 LLM Context 的格式。

    实现策略:
        - XML 标签格式（Claude/GPT-4 推荐）
        - Markdown 格式（通用）
        - 极简格式（最小化 Token）
    """

    @abstractmethod
    def render(
        self,
        results: List,
        render_format: Optional["RenderFormat"] = None
    ) -> str:
        """
        渲染记忆列表为上下文字符串
        
        Args:
            results: SearchResult 列表或 MemoryAtom 列表
            render_format: 输出格式（可选，覆盖默认）
        
        Returns:
            str: 渲染后的上下文字符串
        
        Examples:
            >>> renderer = BaseContextRenderer(format=RenderFormat.XML)
            >>> context = renderer.render(search_results.results)
            >>> print(context)
            <system_memory_context>...
        """
        pass


# ========== 导出列表 ==========

__all__ = [
    # 接口
    "BaseMemoryRetriever",
    "BaseContextRenderer",
    "BaseFusion",
    "BaseReranker",
]
