"""
稀疏向量检索器 (Sparse Retriever)

使用 BGE-M3 的稀疏向量进行词汇级精准匹配检索。
捕获精准实体匹配，如特定函数名、错误码等。

职责:
- 执行稀疏向量检索 (BGE-M3 Sparse/BM25)

对应设计文档: PROJECT.md 5.1.2 节
"""

from typing import Optional
import logging

from hivememory.retrieval.interfaces import MemorySearcher
from hivememory.core.config import SparseRetrieverConfig
from hivememory.retrieval.models import (
    SearchResult,
    SearchResults,
    ProcessedQuery,
)

logger = logging.getLogger(__name__)


class SparseRetriever(MemorySearcher):
    """
    稀疏向量检索器

    使用 BGE-M3 的稀疏向量进行词汇级精准匹配。
    捕获精准实体匹配，如特定函数名、错误码、专有名词等。
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        config: Optional[SparseRetrieverConfig] = None
    ):
        """
        初始化稀疏检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 检索器配置
        """
        self.storage = storage
        self.config = config or SparseRetrieverConfig()

    def retrieve(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行稀疏向量检索

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        import time
        start_time = time.time()

        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        # 构建过滤条件
        filters = query.filters.to_qdrant_filter() if query.filters else None

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.debug(f"Sparse检索: '{search_text[:50]}...', filters={filters}")

        try:
            # 执行稀疏向量检索
            raw_results = self.storage.search_memories(
                query_text=search_text,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
                mode="sparse"
            )
        except Exception as e:
            logger.error(f"Sparse检索失败: {e}")
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)

        # 转换结果
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            sparse_score = hit["score"]

            # 稀疏检索不应用时间衰减，直接使用原始分数
            search_results.append(SearchResult(
                memory=memory,
                score=sparse_score,
                match_reason=f"Sparse: {sparse_score:.3f}"
            ))

        latency = (time.time() - start_time) * 1000
        logger.info(f"Sparse检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results,
            total_candidates=len(raw_results),
            latency_ms=latency
        )

    def search(
        self,
        query: ProcessedQuery,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> SearchResults:
        """
        MemorySearcher 接口实现

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        return self.retrieve(query, top_k, score_threshold)


__all__ = [
    "SparseRetriever",
]
