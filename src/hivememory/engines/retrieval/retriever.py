"""
记忆检索器模块 (MemoryRetrievers)

包含各类记忆检索器的实现：
1. DenseRetriever: 稠密向量检索
2. SparseRetriever: 稀疏向量检索
3. HybridRetriever: 混合检索 (Dense + Sparse + RRF)
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hivememory.config.patchouli import (
    DenseRetrieverConfig,
    HybridRetrieverConfig,
    SparseRetrieverConfig,
)
from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError
from hivememory.engines.retrieval.fusion import create_fusion
from hivememory.engines.retrieval.interfaces import BaseFusion, BaseMemoryRetriever
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    SearchResult,
    SearchResults,
)

if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore


logger = logging.getLogger(__name__)


class DenseRetriever(BaseMemoryRetriever):
    """
    稠密向量检索器

    使用 Qdrant 的稠密向量进行语义检索，捕获模糊语义匹配。
    """

    def __init__(
        self,
        mid_term: "MidTermMemoryStore",
        config: DenseRetrieverConfig,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        """
        初始化稠密检索器

        Args:
            mid_term: 中期记忆存储
            config: 检索器配置
            now: 评分决策时刻的局部注入（A2-P 时间边界）；默认当前 UTC
        """
        self.mid_term = mid_term
        self._now = now or (lambda: datetime.now(UTC))
        self.config = config

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
        score_threshold: float | None = None,
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

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.info(f"Dense检索: '{search_text[:50]}...', filters={query.filters}")

        try:
            raw_results = await self.mid_term.search(
                query.belong_to,
                from_actor=query.from_actor,
                query=search_text,
                top_k=top_k,
                filters=query.filters,
                mode="dense",
                score_threshold=score_threshold,
            )
        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(f"Dense检索失败: {e}", exc_info=True)
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
                # 时间衰减基准与生命力评分一致：decay_anchor_at（A2-P §3.1）。
                decay = self._calculate_time_decay(memory.meta.lifecycle.decay_anchor_at)
                boost = (1 - decay) * 0.1  # 最多 10% 的时间惩罚
                final_score = vector_score * (1 - boost)

            # 置信度加权
            if self.config.enable_confidence_boost:
                confidence_boost = memory.meta.lifecycle.confidence_score * 0.05
                final_score += confidence_boost

            search_results.append(
                SearchResult(
                    memory=memory,
                    score=final_score,
                    vector_score=vector_score,
                    boost_applied=boost,
                    match_reason=f"Dense: {vector_score:.3f}",
                )
            )

        latency = (time.time() - start_time) * 1000
        logger.info(f"Dense检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results, total_candidates=len(raw_results), latency_ms=latency
        )

    def _calculate_time_decay(self, anchor_at: datetime) -> float:
        """
        计算时间衰减系数

        使用指数衰减: decay = e^(-λt)
        其中 λ = ln(2) / half_life_days

        Args:
            anchor_at: 衰减基准时间（``meta.lifecycle.decay_anchor_at``）

        Returns:
            衰减系数 (0-1)，越新越接近 1
        """
        # 决策时刻用注入的 now（aware UTC），不读墙钟
        delta = self._now() - anchor_at
        days_elapsed = delta.total_seconds() / (24 * 3600)

        # 指数衰减
        lambda_val = math.log(2) / self.config.time_decay_days
        decay = math.exp(-lambda_val * days_elapsed)

        return decay


class SparseRetriever(BaseMemoryRetriever):
    """
    稀疏向量检索器

    使用 BGE-M3 的稀疏向量进行词汇级精准匹配。
    捕获精准实体匹配，如特定函数名、错误码、专有名词等。
    """

    def __init__(self, mid_term: "MidTermMemoryStore", config: SparseRetrieverConfig):
        """
        初始化稀疏检索器

        Args:
            mid_term: 中期记忆存储
            config: 检索器配置
        """
        self.mid_term = mid_term
        self.config = config

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
        score_threshold: float | None = None,
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

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.info(f"Sparse检索: '{search_text[:50]}...', filters={query.filters}")

        try:
            raw_results = await self.mid_term.search(
                query.belong_to,
                from_actor=query.from_actor,
                query=search_text,
                top_k=top_k,
                filters=query.filters,
                mode="sparse",
                score_threshold=score_threshold,
            )
        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(f"Sparse检索失败: {e}", exc_info=True)
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)

        # 转换结果
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            sparse_score = hit["score"]

            # 稀疏检索不应用时间衰减，直接使用原始分数
            search_results.append(
                SearchResult(
                    memory=memory, score=sparse_score, match_reason=f"Sparse: {sparse_score:.3f}"
                )
            )

        latency = (time.time() - start_time) * 1000
        logger.info(f"Sparse检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results, total_candidates=len(raw_results), latency_ms=latency
        )


class HybridRetriever(BaseMemoryRetriever):
    """
    混合检索器

    结合稠密向量检索、稀疏向量检索和 RRF 融合，提供高质量的检索结果。

    架构:
        1. Parallel Recall: DenseRetriever || SparseRetriever
        2. Fusion: RRF 合并结果
        3. Return: Top-K 融合结果

    重排序不属于检索器，是 RetrievalEngine 中召回之后的独立阶段。
    """

    def __init__(
        self,
        config: HybridRetrieverConfig,
        dense_retriever: BaseMemoryRetriever,
        sparse_retriever: BaseMemoryRetriever,
        fusion: BaseFusion,
    ):
        """
        初始化混合检索器

        Args:
            config: 混合检索配置
            dense_retriever: 稠密检索器实例；该路在配置中关闭时为 NoOpRetriever
            sparse_retriever: 稀疏检索器实例；该路在配置中关闭时为 NoOpRetriever
            fusion: 融合器实例
        """
        self.config = config
        self.enable_parallel = config.enable_parallel

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.fusion = fusion

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> SearchResults:
        """
        执行混合检索

        执行稠密+稀疏并行召回和 RRF 融合。

        Args:
            query: 处理后的查询
            top_k: 返回数量（可选）
            score_threshold: 相似度阈值（可选）

        Returns:
            检索结果集合
        """
        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        start_time = time.time()

        # 并行召回
        if self.enable_parallel:
            dense_results, sparse_results = await self._parallel_recall(query)
        else:
            dense_results, sparse_results = await self._sequential_recall(query)

        # RRF 融合
        fused_results = self.fusion.fuse(dense_results, sparse_results)

        # 融合分数（RRF 名次分数等）与 cosine 相似度阈值不可比，融合后不套用阈值。
        if score_threshold > 0:
            logger.debug(
                f"跳过融合后阈值过滤 (threshold={score_threshold})：融合分数不适用相似度阈值"
            )

        # 截取 top_k
        fused_results.results = fused_results.results[:top_k]

        latency = (time.time() - start_time) * 1000
        logger.info(f"混合检索完成: 返回 {len(fused_results)} 条结果, 耗时 {latency:.1f}ms")

        return fused_results

    async def _parallel_recall(self, query: RetrievalQuery) -> tuple[SearchResults, SearchResults]:
        """asyncio.gather 并行执行稠密和稀疏检索

        单路的普通失败降级为空结果；存储离线/读取失败与取消等非 ``Exception``
        的 ``BaseException`` 必须向上传播，不能被当作检索结果交给融合。
        """
        dense_outcome, sparse_outcome = await asyncio.gather(
            self.dense_retriever.retrieve(query),
            self.sparse_retriever.retrieve(query),
            return_exceptions=True,
        )
        for outcome in (dense_outcome, sparse_outcome):
            if isinstance(outcome, (StorageOfflineError, StorageReadError)):
                raise outcome
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                raise outcome

        dense_results = (
            SearchResults() if isinstance(dense_outcome, BaseException) else dense_outcome
        )
        sparse_results = (
            SearchResults() if isinstance(sparse_outcome, BaseException) else sparse_outcome
        )
        return dense_results, sparse_results

    async def _sequential_recall(
        self, query: RetrievalQuery
    ) -> tuple[SearchResults, SearchResults]:
        """顺序执行稠密和稀疏检索"""
        dense_results = await self.dense_retriever.retrieve(query)
        sparse_results = await self.sparse_retriever.retrieve(query)
        return dense_results, sparse_results


class NoOpRetriever(BaseMemoryRetriever):
    """
    空召回

    在配置中关闭的召回路径由它顶替，保持检索器接口调用一致；总是返回空结果。
    """

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> SearchResults:
        return SearchResults()


def create_retriever(
    mid_term: "MidTermMemoryStore",
    config: HybridRetrieverConfig | DenseRetrieverConfig | SparseRetrieverConfig,
) -> BaseMemoryRetriever:
    """
    创建检索器工厂

    召回策略按配置多态分发:
    - DenseRetrieverConfig -> DenseRetriever
    - SparseRetrieverConfig -> SparseRetriever
    - HybridRetrieverConfig -> HybridRetriever

    重排序不在此组装，由 RetrievalEngine 作为独立阶段持有。

    Args:
        mid_term: 中期记忆存储
        config: 召回策略配置

    Returns:
        BaseMemoryRetriever 实例
    """
    # 多态分发
    if isinstance(config, DenseRetrieverConfig):
        return DenseRetriever(mid_term, config)

    if isinstance(config, SparseRetrieverConfig):
        return SparseRetriever(mid_term, config)

    if isinstance(config, HybridRetrieverConfig):
        # 1. 创建子检索器；关闭的召回路径由 NoOpRetriever 顶替
        dense_retriever: BaseMemoryRetriever = (
            DenseRetriever(mid_term, config.dense) if config.dense.enabled else NoOpRetriever()
        )
        sparse_retriever: BaseMemoryRetriever = (
            SparseRetriever(mid_term, config.sparse) if config.sparse.enabled else NoOpRetriever()
        )

        # 2. 创建 Fusion
        fusion = create_fusion(config.fusion)

        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
            config=config,
        )

    raise ValueError(f"未知的 Retriever 类型: {type(config)}")


__all__ = [
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "NoOpRetriever",
    "create_retriever",
]
