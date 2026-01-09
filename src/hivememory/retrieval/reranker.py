"""
重排序器接口 (Reranker)

定义结果重排序的抽象接口，用于在 RRF 融合后对结果进行精排。

预留接口:
- NoopReranker: 透传，不做任何处理
- CrossEncoderReranker: (未来实现) 使用 Cross-Encoder 模型进行精排

对应设计文档: PROJECT.md 5.1 节
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List
import logging
import threading

from hivememory.retrieval.interfaces import BaseReranker
from hivememory.retrieval.models import SearchResults, ProcessedQuery
from hivememory.core.config import RerankerConfig

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
        query: ProcessedQuery
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

    使用 BGE-Reranker-v2-m3 模型对 RRF 融合后的结果进行精排。

    特点:
    - 延迟加载模型 (首次调用时加载)
    - 线程安全的模型初始化
    - 批处理优化
    - 优雅的错误降级

    Examples:
        使用 BGE-Reranker-v2-m3:
        >>> from hivememory.retrieval.reranker import CrossEncoderReranker
        >>> from hivememory.core.config import RerankerConfig
        >>> config = RerankerConfig(model_name="BAAI/bge-reranker-v2-m3")
        >>> reranker = CrossEncoderReranker(config)
        >>> reranked = reranker.rerank(results, query)
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        初始化 CrossEncoderReranker

        Args:
            config: Reranker 配置
            device: 运行设备 (覆盖 config.device)
            use_fp16: 是否使用 FP16 (覆盖 config.use_fp16)
        """
        self.config = config or RerankerConfig()
        self.device = device or self.config.device
        self.use_fp16 = use_fp16 if use_fp16 is not None else self.config.use_fp16

        # 延迟加载模型
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()

        logger.info(
            f"CrossEncoderReranker 配置: "
            f"model={self.config.model_name}, device={self.device}"
        )

    @property
    def model(self) -> Any:
        """延迟加载模型，首次使用时才加载"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self) -> None:
        """加载 BGE-Reranker 模型"""
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding 未安装。请运行: pip install FlagEmbedding"
            )

        logger.info(f"正在加载 Reranker 模型: {self.config.model_name}")
        try:
            self._model = FlagReranker(
                model_name_or_path=self.config.model_name,
                device=self.device,
                use_fp16=self.use_fp16,
            )
            logger.info("Reranker 模型加载完成")
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise

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
        query: ProcessedQuery
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
        query_text = query.semantic_query or query.original_query
        pairs = [
            [query_text, r.memory.index.get_embedding_text()]
            for r in candidates
        ]

        # 3. 批量计算 rerank 分数
        try:
            raw_scores = self.model.compute_score(pairs)
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
            query_used=results.query_used,
        )


def create_reranker(config: Optional[RerankerConfig] = None) -> BaseReranker:
    """
    创建 Reranker 实例的工厂函数

    根据配置自动创建合适的 Reranker 实例。

    Args:
        config: Reranker 配置

    Returns:
        NoopReranker 或 CrossEncoderReranker 实例

    Examples:
        >>> from hivememory.retrieval.reranker import create_reranker
        >>> from hivememory.core.config import RerankerConfig
        >>> config = RerankerConfig(type="cross_encoder")
        >>> reranker = create_reranker(config)
        >>> isinstance(reranker, CrossEncoderReranker)
        True
        >>>
        >>> # 创建 NoopReranker
        >>> config = RerankerConfig(enabled=False)
        >>> reranker = create_reranker(config)
        >>> isinstance(reranker, NoopReranker)
        True
    """
    config = config or RerankerConfig()

    if not config.enabled:
        return NoopReranker(config)

    if config.type == "cross_encoder":
        return CrossEncoderReranker(config)

    return NoopReranker(config)


__all__ = [
    "NoopReranker",
    "CrossEncoderReranker",
    "create_reranker",
]
