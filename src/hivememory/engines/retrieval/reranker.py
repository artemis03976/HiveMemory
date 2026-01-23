"""
重排序器接口 (Reranker)

定义结果重排序的抽象接口，用于在 RRF 融合后对结果进行精排。

预留接口:
- NoopReranker: 透传，不做任何处理
- CrossEncoderReranker: 使用 Infrastructure 层服务进行精排

对应设计文档: PROJECT.md 5.1 节
"""

from typing import Optional, List
import logging

from hivememory.engines.retrieval.interfaces import BaseReranker
from hivememory.engines.retrieval.models import SearchResults, RetrievalQuery
from hivememory.patchouli.config import RerankerConfig
from hivememory.infrastructure.rerank.base import BaseRerankService
from hivememory.utils import MemoryAtomRenderer

logger = logging.getLogger(__name__)


class NoopReranker(BaseReranker):
    """
    透传重排序器

    不做任何处理，直接返回原结果。
    作为默认实现和占位符。
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        初始化 NoopReranker

        Args:
            config: 配置
        """
        self.config = config or RerankerConfig()

    def rerank(
        self,
        results: SearchResults,
        query: RetrievalQuery
    ) -> SearchResults:
        """
        透传结果，不做重排序

        Args:
            results: 检索结果
            query: 查询

        Returns:
            原样返回的结果
        """
        logger.debug("NoopReranker: 透传结果，不进行重排序")
        return results


class CrossEncoderReranker(BaseReranker):
    """
    CrossEncoder 重排序器

    使用 Infrastructure 层提供的 RerankService 对 RRF 融合后的结果进行精排。
    """

    def __init__(
        self,
        service: BaseRerankService,
        config: Optional[RerankerConfig] = None,
    ):
        """
        初始化 CrossEncoderReranker

        Args:
            service: Rerank 服务实例
            config: Reranker 配置
        """
        self.service = service
        self.config = config or RerankerConfig()
        
        logger.info("CrossEncoderReranker 初始化完成")

    def _normalize_score(self, score: float) -> float:
        """
        使用 sigmoid 将分数映射到 0-1

        BGE-Reranker 原始输出范围约 -10 到 +10，
        使用 sigmoid 函数将其映射到 0-1 范围以便于阈值过滤。

        Args:
            score: 原始 reranker 分数

        Returns:
            标准化后的分数 (0-1)
        """
        import math
        return 1 / (1 + math.exp(-score))

    def rerank(
        self,
        results: SearchResults,
        query: RetrievalQuery
    ) -> SearchResults:
        """
        使用 Cross-Encoder 模型重排序结果

        处理流程:
        1. 过滤到 top_k 候选 (性能优化)
        2. 构建 [query, passage] 对
        3. 批量计算 rerank 分数
        4. 标准化分数并重新排序
        5. 更新 match_reason

        Args:
            results: RRF 融合后的检索结果
            query: 处理后的查询

        Returns:
            重排序后的结果
        """
        import copy

        # 空结果直接返回
        if results.is_empty():
            logger.debug("CrossEncoderReranker: 结果为空，跳过重排序")
            return results

        # 1. 限制到 top_k 候选 (性能优化)
        candidates = results.results[:self.config.top_k]
        logger.debug(
            f"CrossEncoderReranker: 对 {len(candidates)} 条结果进行重排序 "
            f"(top_k={self.config.top_k})"
        )

        # 2. 构建 [query, passage] 对
        query_text = query.semantic_query
        pairs = [
            [query_text, MemoryAtomRenderer.for_dense_embedding(r.memory)]
            for r in candidates
        ]

        # 3. 批量计算 rerank 分数
        try:
            raw_scores = self.service.compute_score(pairs)
        except Exception as e:
            logger.warning(f"Reranker 计算失败，返回原始结果: {e}")
            return results

        # 4. 标准化分数并更新结果
        reranked_results = []
        for result, raw_score in zip(candidates, raw_scores):
            new_result = copy.copy(result)
            new_result.score = (
                self._normalize_score(raw_score)
                if self.config.normalize_scores
                else raw_score
            )
            new_result.match_reason = (
                f"Rerank (score: {new_result.score:.3f}, "
                f"original: {result.score:.3f})"
            )
            reranked_results.append(new_result)

        # 5. 重新排序 (分数降序)
        reranked_results.sort(key=lambda r: r.score, reverse=True)

        # 6. 构建新的 SearchResults
        return SearchResults(
            results=reranked_results,
            total_candidates=results.total_candidates,
            latency_ms=results.latency_ms,
        )


def create_reranker(
    config: Optional[RerankerConfig] = None,
    service: Optional[BaseRerankService] = None
) -> BaseReranker:
    """
    创建 Reranker 实例的工厂函数

    根据配置自动创建合适的 Reranker 实例。

    Args:
        config: Reranker 配置
        service: Rerank 服务实例 (仅当 type=cross_encoder 时需要)

    Returns:
        NoopReranker 或 CrossEncoderReranker 实例
    """
    config = config or RerankerConfig()

    if not config.enabled:
        return NoopReranker(config)

    if config.type == "cross_encoder":
        if service is None:
            logger.warning("请求创建 CrossEncoderReranker 但未提供 service，降级为 NoopReranker")
            return NoopReranker(config)
        return CrossEncoderReranker(service, config)

    return NoopReranker(config)


__all__ = [
    "NoopReranker",
    "CrossEncoderReranker",
    "create_reranker",
]
