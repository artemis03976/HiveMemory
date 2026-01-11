"""
基础检索器模块 (Base Retrievers)

包含基本的稠密和稀疏向量检索器实现。
"""

import logging
import math
import time
from datetime import datetime
from typing import Optional

from hivememory.core.config import DenseRetrieverConfig, SparseRetrieverConfig
from hivememory.retrieval.interfaces import MemorySearcher
from hivememory.retrieval.models import (
    ProcessedQuery,
    SearchResult,
    SearchResults,
)

logger = logging.getLogger(__name__)


class DenseRetriever(MemorySearcher):
    """
    稠密向量检索器

    使用 Qdrant 的稠密向量进行语义检索，捕获模糊语义匹配。
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        config: Optional[DenseRetrieverConfig] = None
    ):
        """
        初始化稠密检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 检索器配置
        """
        self.storage = storage
        self.config = config or DenseRetrieverConfig()

    def retrieve(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行稠密向量检索

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        start_time = time.time()

        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        # 构建过滤条件
        filters = query.filters.to_qdrant_filter() if query.filters else None

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.debug(f"Dense检索: '{search_text[:50]}...', filters={filters}")

        try:
            # 执行稠密向量检索
            raw_results = self.storage.search_memories(
                query_text=search_text,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters if filters else None,
                mode="dense"
            )
        except Exception as e:
            logger.error(f"Dense检索失败: {e}")
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)

        # 转换结果并应用加权
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            vector_score = hit["score"]

            # 计算最终分数
            final_score = vector_score
            boost = 0.0

            # 时间衰减
            if self.config.enable_time_decay:
                decay = self._calculate_time_decay(memory.meta.updated_at)
                boost = (1 - decay) * 0.1  # 最多 10% 的时间惩罚
                final_score = vector_score * (1 - boost)

            # 置信度加权
            if self.config.enable_confidence_boost:
                confidence_boost = memory.meta.confidence_score * 0.05
                final_score += confidence_boost

            search_results.append(SearchResult(
                memory=memory,
                score=final_score,
                vector_score=vector_score,
                boost_applied=boost,
                match_reason=f"Dense: {vector_score:.3f}"
            ))

        latency = (time.time() - start_time) * 1000
        logger.info(f"Dense检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results,
            total_candidates=len(raw_results),
            latency_ms=latency
        )

    def search(
        self,
        query: ProcessedQuery,
        top_k: int = 5,
        score_threshold: float = 0.75
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

    def _calculate_time_decay(self, updated_at: datetime) -> float:
        """
        计算时间衰减系数

        使用指数衰减: decay = e^(-λt)
        其中 λ = ln(2) / half_life_days

        Args:
            updated_at: 更新时间

        Returns:
            衰减系数 (0-1)，越新越接近 1
        """
        now = datetime.now()
        delta = now - updated_at
        days_elapsed = delta.total_seconds() / (24 * 3600)

        # 指数衰减
        lambda_val = math.log(2) / self.config.time_decay_days
        decay = math.exp(-lambda_val * days_elapsed)

        return decay


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
