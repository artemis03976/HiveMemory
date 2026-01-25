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

from hivememory.patchouli.config import FusionConfig, AdaptiveWeightedFusionConfig, RetrievalModeConfig
from hivememory.engines.retrieval.models import SearchResult, SearchResults
from hivememory.engines.retrieval.interfaces import BaseFusion

logger = logging.getLogger(__name__)


class ReciprocalRankFusion(BaseFusion):
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


class AdaptiveWeightedFusion(BaseFusion):
    """
    自适应加权融合器

    核心算法:
        S_final = Σ(w_i × S_i) × M(C, V)

    其中:
        - 左侧 (动态相关性): 根据检索模式动态分配权重
        - 右侧 (质量乘数): 基于 Confidence 和 Vitality 的惩罚/奖励因子

    支持的检索模式:
        - debug: 高 sparse 权重，强置信度惩罚 (精确匹配场景)
        - concept: 高 dense 权重，弱惩罚 (概念理解场景)
        - timeline: 高 time 权重，中等惩罚 (时间相关场景)
        - brainstorm: 高 dense 权重，无惩罚 (发散思维场景)
    """

    def __init__(self, config: Optional[AdaptiveWeightedFusionConfig] = None):
        """
        初始化自适应加权融合器

        Args:
            config: 融合配置，包含各模式的权重和质量乘数参数
        """
        self.config = config or AdaptiveWeightedFusionConfig()

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        mode: Optional[str] = None
    ) -> SearchResults:
        """
        使用自适应加权算法融合检索结果

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            mode: 检索模式 (debug/concept/timeline/brainstorm)，None 则使用默认模式

        Returns:
            融合后的结果集合
        """
        # 获取模式配置
        mode = mode or self.config.default_mode
        mode_config = self._get_mode_config(mode)

        # 分数累加器
        scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        # 计算权重归一化因子
        total_weight = mode_config.dense_weight + mode_config.sparse_weight
        if total_weight == 0:
            total_weight = 1.0

        # 处理稠密检索结果
        for result in dense_results.results:
            memory_id = str(result.memory.id)
            # 使用原始分数进行加权
            weighted_score = (mode_config.dense_weight / total_weight) * result.score
            scores[memory_id] += weighted_score
            result_map[memory_id] = result

        # 处理稀疏检索结果
        for result in sparse_results.results:
            memory_id = str(result.memory.id)
            weighted_score = (mode_config.sparse_weight / total_weight) * result.score
            scores[memory_id] += weighted_score

            if memory_id not in result_map:
                result_map[memory_id] = result
            else:
                # 合并匹配原因
                existing = result_map[memory_id]
                new_result = result.model_copy()
                new_result.match_reason = f"{existing.match_reason} | {result.match_reason}"
                result_map[memory_id] = new_result

        # 应用质量乘数
        for memory_id in scores:
            result = result_map[memory_id]
            confidence = result.memory.meta.confidence_score
            vitality = result.memory.meta.vitality_score

            quality_multiplier = self._calculate_quality_multiplier(
                confidence, vitality, mode_config
            )
            scores[memory_id] *= quality_multiplier

        # 按最终分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建最终结果
        fused_results = []
        for memory_id in sorted_ids[:self.config.final_top_k]:
            result = result_map[memory_id]
            # 创建新的 SearchResult 以避免修改原始对象
            new_result = result.model_copy()
            new_result.score = scores[memory_id]
            fused_results.append(new_result)

        logger.info(
            f"AdaptiveWeightedFusion[{mode}]: Dense({len(dense_results)}) + Sparse({len(sparse_results)}) "
            f"-> Fused({len(fused_results)}), 唯一记忆={len(result_map)}"
        )

        total_latency = dense_results.latency_ms + sparse_results.latency_ms

        return SearchResults(
            results=fused_results,
            total_candidates=len(result_map),
            latency_ms=total_latency
        )

    def fuse_with_intent(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        query_intent: str
    ) -> SearchResults:
        """
        基于意图自动选择模式进行融合 (预留接口)

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            query_intent: Gateway 传入的查询意图

        Returns:
            融合后的结果集合
        """
        mode = self._infer_mode_from_intent(query_intent)
        return self.fuse(dense_results, sparse_results, mode)

    def _infer_mode_from_intent(self, intent: str) -> str:
        """
        从意图推断检索模式 (预留实现)

        未来可接入 LLM 或规则引擎进行更智能的模式选择。

        Args:
            intent: 查询意图

        Returns:
            检索模式名称
        """
        # 简单的规则映射 (预留扩展)
        intent_mode_map = {
            "debug": "debug",
            "fix": "debug",
            "error": "debug",
            "concept": "concept",
            "explain": "concept",
            "how": "concept",
            "timeline": "timeline",
            "when": "timeline",
            "history": "timeline",
            "brainstorm": "brainstorm",
            "idea": "brainstorm",
        }

        intent_lower = intent.lower()
        for keyword, mode in intent_mode_map.items():
            if keyword in intent_lower:
                return mode

        return self.config.default_mode

    def _get_mode_config(self, mode: str) -> RetrievalModeConfig:
        """
        获取指定模式的配置

        Args:
            mode: 模式名称

        Returns:
            模式配置对象
        """
        mode_map = {
            "debug": self.config.debug_mode,
            "concept": self.config.concept_mode,
            "timeline": self.config.timeline_mode,
            "brainstorm": self.config.brainstorm_mode,
        }

        if mode not in mode_map:
            logger.warning(f"未知的检索模式: {mode}, 使用默认模式: {self.config.default_mode}")
            mode = self.config.default_mode

        return mode_map.get(mode, self.config.concept_mode)

    def _calculate_quality_multiplier(
        self,
        confidence: float,
        vitality: float,
        mode_config: RetrievalModeConfig
    ) -> float:
        """
        计算质量乘数 M(C, V) = Factor_conf × Factor_vit

        Args:
            confidence: 置信度分数 (0-1)
            vitality: 生命力分数 (0-100)
            mode_config: 模式配置

        Returns:
            质量乘数
        """
        # 置信度因子
        conf_factor = 1.0
        if mode_config.confidence_penalty_enabled:
            if confidence < mode_config.confidence_penalty_threshold:
                conf_factor = mode_config.confidence_penalty_factor

        # 生命力因子
        vit_factor = 1.0
        if mode_config.vitality_boost_enabled:
            if vitality >= mode_config.vitality_high_threshold:
                vit_factor = mode_config.vitality_high_factor
            elif vitality <= mode_config.vitality_low_threshold:
                vit_factor = mode_config.vitality_low_factor

        return conf_factor * vit_factor


__all__ = [
    "ReciprocalRankFusion",
    "AdaptiveWeightedFusion",
]
