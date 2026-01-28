"""
HiveMemory 语义边界吸附器

无状态服务，基于本地 Embedding 模型实现智能话题分割。

核心算法 (v3.0 无状态版):
    1. 启发式强吸附（停用词检测）
    2. 双阈值向量筛选（高/低阈值）
    3. 灰度区间智能仲裁（Reranker/SLM）

参考: PROJECT.md 4.1.2 节, docs/mod/DiscourseContinuity.md

作者: HiveMemory Team
版本: 3.0.0
"""


import logging
from typing import List, Optional

from hivememory.patchouli.config import SemanticAdsorberConfig
from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    LogicalBlock,
    SemanticBuffer,
)
from hivememory.engines.perception.interfaces import BaseArbiter
from hivememory.engines.perception.context_bridge import ContextBridge
from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.infrastructure.rerank.base import BaseRerankService

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


class SemanticBoundaryAdsorber:
    """
    语义边界吸附器

    无状态服务，实现三阶段处理管道：
        Step 1: 启发式强吸附（停用词检测）
        Step 2: 向量筛选（双阈值：0.75 高 / 0.40 低）
        Step 3: 智能仲裁（灰度区间使用 Reranker/SLM）

    特性：
        - 无状态：不维护内部状态，所有状态通过参数传入
        - 返回 FlushEvent：统一的决策输出格式
        - 纯函数：compute_new_topic_kernel() 不修改 buffer

    Examples:
        >>> adsorber = SemanticBoundaryAdsorber(
        ...     config=SemanticAdsorberConfig(),
        ...     embedding_service=embedding_service,
        ...     context_bridge=context_bridge,
        ...     arbiter=arbiter
        ... )
        >>> flush_event = adsorber.should_adsorb(buffer, new_block)
        >>> if flush_event:
        ...     # 需要 flush
        ...     handle_flush(flush_event)
    """

    def __init__(
        self,
        config: SemanticAdsorberConfig,
        embedding_service: BaseEmbeddingService = None,
        context_bridge: Optional[ContextBridge] = None,
        arbiter: Optional[BaseArbiter] = None,
    ):
        """
        初始化语义边界吸附器

        Args:
            config: 配置对象
            embedding_service: Embedding 服务实例
            context_bridge: 上下文桥接器
            arbiter: 灰度仲裁器
        """
        self.config = config or SemanticAdsorberConfig()
        self.embedding_service = embedding_service
        self.context_bridge = context_bridge
        self.arbiter = arbiter

        logger.info(
            f"SemanticBoundaryAdsorber 初始化: "
            f"threshold_high={self.config.semantic_threshold_high}, "
            f"threshold_low={self.config.semantic_threshold_low}, "
        )

    def should_adsorb(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> Optional[FlushEvent]:
        """
        判断是否需要 flush（统一处理管道）

        判定流程：
            Step 1: 启发式强吸附（停用词检测）
            Step 2: 向量筛选（双阈值）
            Step 3: 智能仲裁（灰度区间）

        Args:
            buffer: 当前语义缓冲区（只读）
            new_block: 新的 LogicalBlock

        Returns:
            None: 继续吸附，不需要 flush
            FlushEvent: 需要 flush，包含原因和要刷出的 blocks
        """
        # ========== Step 1: 启发式强吸附 ==========
        # 检查是否为非信息性短文本
        if self._is_non_informative_short_text(new_block):
            logger.debug("启发式强吸附: 非信息性短文本")
            return None  # 继续吸附

        # ========== Step 2: 向量筛选（双阈值） ==========
        if buffer.topic_kernel_vector is not None:
            # 使用 ContextBridge 构建增强的锚点文本
            anchor_text = self._build_enhanced_anchor(new_block, buffer)

            if not anchor_text:
                # 无法构建锚点，默认吸附
                logger.debug("锚点文本为空，默认吸附")
                return None

            # 计算相似度
            similarity = 0.0
            try:
                text_vector = self.embedding_service.encode(anchor_text, normalize=True)
                similarity = self.embedding_service.compute_cosine_similarity(
                    text_vector,
                    buffer.topic_kernel_vector
                )
            except Exception as e:
                logger.warning(f"计算相似度时出错: {e}")

            logger.debug(
                f"语义相似度: {similarity:.3f} "
                f"(低阈值: {self.config.semantic_threshold_low}, "
                f"高阈值: {self.config.semantic_threshold_high})"
            )

            # 高阈值: 吸附
            if similarity >= self.config.semantic_threshold_high:
                logger.debug(f"高阈值判定: {similarity:.3f} >= {self.config.semantic_threshold_high} -> ADSORB")
                return None

            # 低阈值: 强制切分
            if similarity < self.config.semantic_threshold_low:
                logger.debug(f"低阈值判定: {similarity:.3f} < {self.config.semantic_threshold_low} -> SPLIT")
                return FlushEvent(
                    flush_reason=FlushReason.SEMANTIC_DRIFT,
                    blocks_to_flush=buffer.blocks.copy(),
                    triggered_by_block=new_block,
                )

            # ========== Step 3: 智能仲裁（灰度区间） ==========
            # low <= similarity < high
            if self.config.arbiter.enabled and self.arbiter.is_available():
                return self._arbitrate_grey_area(buffer, new_block, similarity)
            else:
                # 仲裁器不可用或未启用，使用保守策略：继续吸附
                logger.debug("灰度区间默认吸附 (仲裁器未启用或不可用)")
                return None

        # 默认吸附（无话题核心）
        return None

    def compute_new_topic_kernel(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> Optional[List[float]]:
        """
        计算新的话题核心向量（纯函数）

        策略：指数移动平均 (EMA)
            new_kernel = alpha * new_vector + (1 - alpha) * old_kernel

        注意：此方法不修改 buffer，只返回计算结果。
        调用方负责通过 BufferManager.update_buffer_metadata() 更新 buffer。

        Args:
            buffer: 当前语义缓冲区（只读）
            new_block: 新的 LogicalBlock

        Returns:
            新的话题核心向量，或 None（如果无法计算）
        """
        if not new_block.anchor_text:
            return None

        try:
            # 使用增强的锚点文本进行编码
            anchor_text = self._build_enhanced_anchor(new_block, buffer)
            if not anchor_text:
                anchor_text = new_block.anchor_text

            new_vector = self.embedding_service.encode(anchor_text, normalize=True)

            if buffer.topic_kernel_vector is None:
                # 首次设置话题核心
                logger.debug("话题核心向量初始化")
                return new_vector
            else:
                # EMA 更新
                import numpy as np
                old_vec = np.array(buffer.topic_kernel_vector)
                new_vec = np.array(new_vector)
                updated_kernel = (
                    self.config.ema_alpha * new_vec + (1 - self.config.ema_alpha) * old_vec
                ).tolist()
                logger.debug("话题核心向量已计算（EMA）")
                return updated_kernel

        except Exception as e:
            logger.warning(f"计算话题核心向量失败: {e}")
            return None

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
        if self.context_bridge:
            return self.context_bridge.build_anchor_text(rewritten_query, buffer)
        return rewritten_query

    def _arbitrate_grey_area(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock,
        similarity: float
    ) -> Optional[FlushEvent]:
        """
        灰度区间智能仲裁

        使用 Arbiter 进行更精细的判断。

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock
            similarity: 语义相似度分数

        Returns:
            None: 继续吸附
            FlushEvent: 需要 flush
        """
        # 提取上下文和查询
        previous_context = ""
        if self.context_bridge:
            previous_context = self.context_bridge.extract_last_context(buffer)
        current_query = new_block.rewritten_query or new_block.anchor_text

        if not current_query:
            logger.debug("当前查询为空，默认吸附")
            return None

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
                return None
            else:
                return FlushEvent(
                    flush_reason=FlushReason.SEMANTIC_DRIFT,
                    blocks_to_flush=buffer.blocks.copy(),
                    triggered_by_block=new_block,
                )

        except Exception as e:
            logger.error(f"仲裁失败: {e}，默认吸附")
            return None


def create_adsorber(
    config: SemanticAdsorberConfig,
    embedding_service: BaseEmbeddingService,
    reranker_service: BaseRerankService,
) -> SemanticBoundaryAdsorber:
    """
    创建语义边界吸附器实例

    Args:
        config: 吸附器配置
        embedding_service: 嵌入服务
        reranker_service: 重排服务

    Returns:
        SemanticBoundaryAdsorber: 吸附器实例
    """
    from hivememory.engines.perception.grey_area_arbiter import create_arbiter
    arbiter: BaseArbiter = create_arbiter(
        config.arbiter, 
        reranker_service=reranker_service
    )

    from hivememory.engines.perception.context_bridge import ContextBridge
    context_bridge = ContextBridge(
        config=config.context_bridge,
    )

    return SemanticBoundaryAdsorber(
        config=config,
        embedding_service=embedding_service,
        arbiter=arbiter,
        context_bridge=context_bridge,
    )


__all__ = [
    "SemanticBoundaryAdsorber",
]
