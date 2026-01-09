"""
结果融合模块 (Fusion)

使用 RRF (Reciprocal Rank Fusion) 算法合并多路检索结果。

RRF 公式:
    score(d) = sum(w_i / (k + rank_i(d)))

其中:
    d: 文档
    w_i: 结果列表 i 的权重
    rank_i(d): 文档 d 在列表 i 中的排名
    k: 常数 (通常为 60)

参考: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

对应设计文档: PROJECT.md 5.1.5 节
"""

from typing import Optional, Dict, List
from collections import defaultdict
import logging

from hivememory.core.config import FusionConfig
from hivememory.retrieval.models import SearchResult, SearchResults

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    倒数排名融合 (RRF)

    用于合并稠密和稀疏两路检索结果。
    RRF 对分数分布不敏感，能很好地融合不同检索方式的结果。
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        初始化 RRF 融合器

        Args:
            config: 融合配置
        """
        self.config = config or FusionConfig()

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults
    ) -> SearchResults:
        """
        使用 RRF 融合稠密和稀疏检索结果

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果

        Returns:
            融合后的结果集合
        """
        # 分数累加器
        scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        # 处理稠密检索结果
        for rank, result in enumerate(dense_results.results, start=1):
            memory_id = str(result.memory.id)
            rrf_score = self.config.dense_weight / (self.config.rrf_k + rank)
            scores[memory_id] += rrf_score
            result_map[memory_id] = result

        # 处理稀疏检索结果
        for rank, result in enumerate(sparse_results.results, start=1):
            memory_id = str(result.memory.id)
            rrf_score = self.config.sparse_weight / (self.config.rrf_k + rank)
            scores[memory_id] += rrf_score

            if memory_id not in result_map:
                result_map[memory_id] = result
            else:
                # 合并匹配原因
                existing = result_map[memory_id]
                result.match_reason = f"{existing.match_reason} | {result.match_reason}"
                result_map[memory_id] = result

        # 按 RRF 分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建最终结果
        fused_results = []
        for memory_id in sorted_ids[:self.config.final_top_k]:
            result = result_map[memory_id]
            result.score = scores[memory_id]  # 更新为 RRF 分数
            fused_results.append(result)

        logger.info(
            f"RRF融合: Dense({len(dense_results)}) + Sparse({len(sparse_results)}) "
            f"-> Fused({len(fused_results)}), 唯一记忆={len(result_map)}"
        )

        total_latency = dense_results.latency_ms + sparse_results.latency_ms

        return SearchResults(
            results=fused_results,
            total_candidates=len(result_map),
            latency_ms=total_latency
        )

    def fuse_multi(
        self,
        result_lists: List[SearchResults],
        weights: Optional[List[float]] = None
    ) -> SearchResults:
        """
        融合多路检索结果 (通用接口)

        Args:
            result_lists: 多路检索结果列表
            weights: 每路结果的权重 (可选)

        Returns:
            融合后的结果集合
        """
        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(result_lists) != len(weights):
            raise ValueError("结果列表数量与权重数量不匹配")

        # 分数累加器
        scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        # 处理每一路结果
        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results.results, start=1):
                memory_id = str(result.memory.id)
                rrf_score = weight / (self.config.rrf_k + rank)
                scores[memory_id] += rrf_score

                if memory_id not in result_map:
                    result_map[memory_id] = result

        # 按 RRF 分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建最终结果
        fused_results = []
        for memory_id in sorted_ids[:self.config.final_top_k]:
            result = result_map[memory_id]
            result.score = scores[memory_id]
            fused_results.append(result)

        total_latency = sum(r.latency_ms for r in result_lists)

        return SearchResults(
            results=fused_results,
            total_candidates=len(result_map),
            latency_ms=total_latency
        )


__all__ = [
    "ReciprocalRankFusion",
]
