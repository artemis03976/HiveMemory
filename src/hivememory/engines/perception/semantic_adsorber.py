"""
HiveMemory 语义边界吸附器

基于本地 Embedding 模型实现智能话题分割。

核心算法 (v2.0 增强版):
    1. 启发式强吸附（停用词检测）
    2. 双阈值向量筛选（高/低阈值）
    3. 灰度区间智能仲裁（Reranker/SLM）

参考: PROJECT.md 4.1.2 节, docs/mod/DiscourseContinuity.md

作者: HiveMemory Team
版本: 2.0.0
"""

import logging
from typing import List, Optional, Tuple

from hivememory.core.models import FlushReason
from hivememory.engines.perception.interfaces import SemanticAdsorber, GreyAreaArbiter
from hivememory.engines.perception.models import (
    LogicalBlock,
    SemanticBuffer,
)
from hivememory.engines.perception.context_bridge import (
    ContextBridge,
)
from hivememory.patchouli.config import SemanticAdsorberConfig
from hivememory.infrastructure.embedding.base import BaseEmbeddingService

logger = logging.getLogger(__name__)


# 默认的短文本停用词列表（用于启发式强吸附）
DEFAULT_SHORT_TEXT_STOP_WORDS = {
    # 中文
    "不对", "报错了", "错了", "错误", "嗯", "哦", "啊",
    "好", "行", "可以", "继续", "好的", "然后呢",
    # 英文
    "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "continue", "go on", "sure", "alright", "next",
}


class SemanticBoundaryAdsorber(SemanticAdsorber):
    """
    语义边界吸附器 (v2.0 增强版)

    实现三阶段处理管道：
        Step 1: 启发式强吸附（停用词检测）
        Step 2: 向量筛选（双阈值：0.75 高 / 0.40 低）
        Step 3: 智能仲裁（灰度区间使用 Reranker/SLM）

    新增功能：
        - ContextBridge: 构建增强的语义锚点（包含上一轮上下文）
        - 双阈值系统: 高阈值强吸附，低阈值强制切分
        - 灰度仲裁: 对模糊情况使用 Reranker 进行精确判断
        - 停用词检测: 非信息性短文本强吸附

    Examples:
        >>> from hivememory.engines.perception import ContextBridge, RerankerArbiter
        >>> context_bridge = ContextBridge()
        >>> arbiter = RerankerArbiter(reranker_service)
        >>> adsorber = SemanticBoundaryAdsorber(
        ...     config=SemanticAdsorberConfig(),
        ...     context_bridge=context_bridge,
        ...     arbiter=arbiter
        ... )
        >>> should_adsorb, reason = adsorber.should_adsorb(new_block, buffer)
    """

    def __init__(
        self,
        config: Optional[SemanticAdsorberConfig] = None,
        embedding_service: BaseEmbeddingService = None,
        context_bridge: Optional[ContextBridge] = None,
        arbiter: Optional[GreyAreaArbiter] = None,
    ):
        """
        初始化语义边界吸附器

        Args:
            config: 配置对象
            embedding_service: Embedding 服务实例（推荐通过依赖注入传入）
            context_bridge: 上下文桥接器（默认创建新实例）
            arbiter: 灰度仲裁器（可选）
        """
        self.config = config or SemanticAdsorberConfig()
        self.embedding_service = embedding_service

        # 文本到向量的映射（用于维护话题核心）
        self._anchor_cache: dict = {}

        # 初始化 ContextBridge
        self.context_bridge = context_bridge or ContextBridge()

        # 初始化 Arbiter（可选）
        self.arbiter = arbiter

        logger.info(
            f"SemanticBoundaryAdsorber v2.0 初始化: "
            f"threshold_high={self.config.semantic_threshold_high}, "
            f"threshold_low={self.config.semantic_threshold_low}, "
            f"enable_arbiter={self.config.enable_arbiter}, "
            f"arbiter={'enabled' if arbiter else 'disabled'}, "
            f"embedding_service={'DI' if embedding_service else 'lazy'}"
        )

    def compute_similarity(
        self,
        anchor_text: str,
        topic_kernel: Optional[List[float]]
    ) -> float:
        """
        计算语义相似度

        如果 topic_kernel 为 None，返回 0（强制吸附）

        Args:
            anchor_text: 锚点文本（新 Block 的语义锚点）
            topic_kernel: 话题核心向量

        Returns:
            float: 相似度 (0-1)
        """
        import numpy as np

        if not topic_kernel or not anchor_text:
            return 0.0

        try:
            text_vector = self.embedding_service.encode(anchor_text, normalize=True)
            ref_array = np.array(topic_kernel)
            text_array = np.array(text_vector)
            return float(np.dot(ref_array, text_array))
        except Exception as e:
            logger.warning(f"Embedding 计算失败: {e}")
            return 0.0

    def should_adsorb(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer
    ) -> Tuple[bool, Optional[FlushReason]]:
        """
        判断是否吸附（统一处理管道）

        判定流程：
            Step 1: 启发式强吸附（停用词检测）
            Step 2: 向量筛选（双阈值）
            Step 3: 智能仲裁（灰度区间）

        Args:
            new_block: 新的 LogicalBlock
            buffer: 当前语义缓冲区

        Returns:
            Tuple[bool, Optional[FlushReason]]:
                - 是否吸附
                - 漂移原因（如果不吸附）
        """
        # Block 未闭合，继续等待
        if not new_block.is_complete:
            logger.debug("Block 未闭合，继续等待")
            return True, None

        # ========== Step 1: 启发式强吸附 ==========
        # 检查是否为非信息性短文本
        if self._is_non_informative_short_text(new_block):
            logger.debug("启发式强吸附: 非信息性短文本")
            return True, FlushReason.SHORT_TEXT_ADSORB

        # ========== Step 2: 向量筛选（双阈值） ==========
        if buffer.topic_kernel_vector is not None:
            # 使用 ContextBridge 构建增强的锚点文本
            anchor_text = self._build_enhanced_anchor(new_block, buffer)

            if not anchor_text:
                # 无法构建锚点，默认吸附
                logger.debug("锚点文本为空，默认吸附")
                return True, None

            # 计算相似度
            similarity = self.compute_similarity(
                anchor_text,
                buffer.topic_kernel_vector
            )

            logger.debug(
                f"语义相似度: {similarity:.3f} "
                f"(低阈值: {self.config.semantic_threshold_low}, "
                f"高阈值: {self.config.semantic_threshold_high})"
            )

            # 高阈值: 强吸附
            if similarity >= self.config.semantic_threshold_high:
                logger.debug(f"高阈值判定: {similarity:.3f} >= {self.config.semantic_threshold_high} -> ADSORB")
                return True, None

            # 低阈值: 强制切分
            if similarity < self.config.semantic_threshold_low:
                logger.debug(f"低阈值判定: {similarity:.3f} < {self.config.semantic_threshold_low} -> SPLIT")
                return False, FlushReason.SEMANTIC_DRIFT

            # ========== Step 3: 智能仲裁（灰度区间） ==========
            # low <= similarity < high
            if self.config.enable_arbiter and self.arbiter and self.arbiter.is_available():
                return self._arbitrate_grey_area(new_block, buffer, similarity)
            else:
                # 仲裁器不可用或未启用，使用保守策略：继续吸附
                logger.debug("灰度区间默认吸附 (仲裁器未启用或不可用)")
                return True, None

        # 默认吸附（无话题核心）
        return True, None

    def _is_non_informative_short_text(self, block: LogicalBlock) -> bool:
        """
        判断是否为非信息性短文本（启发式强吸附）

        判定规则：
            1. Token 数小于阈值
            2. 文本内容为停用词

        Args:
            block: 待判定的 LogicalBlock

        Returns:
            bool: 是否为非信息性短文本
        """
        # 获取查询文本
        query_text = block.anchor_text or ""

        # 检查 token 数
        if block.total_tokens >= self.config.short_text_threshold:
            return False

        # 检查是否为停用词
        normalized = query_text.strip().lower()
        stop_words = self.config.stop_words or DEFAULT_SHORT_TEXT_STOP_WORDS
        is_stop_word = normalized in stop_words

        if is_stop_word:
            logger.debug(f"检测到停用词: {query_text}")

        return is_stop_word

    def _build_enhanced_anchor(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer
    ) -> str:
        """
        构建增强的语义锚点文本

        使用 ContextBridge 将上一轮上下文与当前查询组合。

        Args:
            new_block: 新的 LogicalBlock
            buffer: 当前语义缓冲区

        Returns:
            str: 增强的锚点文本
        """
        # 获取 rewritten_query（优先）或原始查询
        rewritten_query = new_block.rewritten_query or new_block.anchor_text

        if not rewritten_query:
            return ""

        # 使用 ContextBridge 构建增强锚点
        return self.context_bridge.build_anchor_text(rewritten_query, buffer)

    def _arbitrate_grey_area(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer,
        similarity: float
    ) -> Tuple[bool, Optional[FlushReason]]:
        """
        灰度区间智能仲裁

        使用 Arbiter 进行更精细的判断。

        Args:
            new_block: 新的 LogicalBlock
            buffer: 当前语义缓冲区
            similarity: 语义相似度分数

        Returns:
            Tuple[bool, Optional[FlushReason]]: 吸附决策
        """
        if not self.arbiter:
            logger.warning("仲裁器未配置，默认吸附")
            return True, None

        # 提取上下文和查询
        previous_context = self.context_bridge.extract_last_context(buffer)
        current_query = new_block.rewritten_query or new_block.anchor_text

        if not current_query:
            logger.debug("当前查询为空，默认吸附")
            return True, None

        try:
            # 调用仲裁器
            should_continue = self.arbiter.should_continue_topic(
                previous_context=previous_context,
                current_query=current_query,
                similarity_score=similarity,
            )

            logger.debug(
                f"灰度仲裁: similarity={similarity:.3f} -> "
                f"{'ADSORB' if should_continue else 'SPLIT'}"
            )

            if should_continue:
                return True, None
            else:
                return False, FlushReason.SEMANTIC_DRIFT

        except Exception as e:
            logger.error(f"仲裁失败: {e}，默认吸附")
            return True, None

    def update_topic_kernel(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> None:
        """
        更新话题核心向量

        策略：指数移动平均 (EMA)
            new_kernel = alpha * new_vector + (1 - alpha) * old_kernel

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock
        """
        if not new_block.anchor_text:
            return

        try:
            # 使用增强的锚点文本进行编码
            anchor_text = self._build_enhanced_anchor(new_block, buffer)
            if not anchor_text:
                anchor_text = new_block.anchor_text

            new_vector = self.embedding_service.encode(anchor_text, normalize=True)

            # 缓存锚点文本
            cache_key = f"{buffer.buffer_id}:{new_block.block_id}"
            self._anchor_cache[cache_key] = {
                "text": anchor_text,
                "vector": new_vector
            }

            if buffer.topic_kernel_vector is None:
                # 首次设置话题核心
                buffer.topic_kernel_vector = new_vector
                logger.debug("话题核心向量初始化")
            else:
                # EMA 更新
                import numpy as np
                old_vec = np.array(buffer.topic_kernel_vector)
                new_vec = np.array(new_vector)
                updated_kernel = (
                    self.config.ema_alpha * new_vec + (1 - self.config.ema_alpha) * old_vec
                ).tolist()
                buffer.topic_kernel_vector = updated_kernel
                logger.debug("话题核心向量已更新（EMA）")

        except Exception as e:
            logger.warning(f"更新话题核心向量失败: {e}")

    def get_topic_summary(self, buffer: SemanticBuffer) -> str:
        """
        获取话题摘要

        Args:
            buffer: 语义缓冲区

        Returns:
            str: 话题摘要
        """
        return buffer.get_topic_summary()

    def reset_topic_kernel(self, buffer: SemanticBuffer) -> None:
        """
        重置话题核心向量

        用于话题切换后清空旧的核心。

        Args:
            buffer: 语义缓冲区
        """
        buffer.topic_kernel_vector = None
        logger.debug("话题核心向量已重置")


__all__ = [
    "SemanticBoundaryAdsorber",
    "DEFAULT_SHORT_TEXT_STOP_WORDS",
]
