"""
HiveMemory 语义边界吸附器

基于本地 Embedding 模型实现智能话题分割。

核心算法：
    1. 短文本强吸附（< 阈值 tokens）
    2. 语义相似度判定（余弦相似度）

注意：
    - Token 溢出检测由 RelayController 负责
    - 空闲超时检测由 IdleTimeoutMonitor 负责

参考: PROJECT.md 4.1.2 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from typing import List, Optional, Tuple

from hivememory.core.models import FlushReason
from hivememory.perception.interfaces import SemanticAdsorber
from hivememory.perception.models import (
    LogicalBlock,
    SemanticBuffer,
)

logger = logging.getLogger(__name__)


class SemanticBoundaryAdsorber(SemanticAdsorber):
    """
    语义边界吸附器

    实现：
        - 基于余弦相似度的语义吸附判定
        - 短文本强吸附策略
        - 锚点对齐策略（仅使用 User Query）
        - 指数移动平均（EMA）更新话题核心

    Examples:
        >>> adsorber = SemanticBoundaryAdsorber()
        >>> should_adsorb, reason = adsorber.should_adsorb(new_block, buffer)
        >>> if not should_adsorb:
        ...     print(f"触发 Flush: {reason}")
    """

    def __init__(
        self,
        semantic_threshold: float = 0.6,
        short_text_threshold: int = 50,  # tokens
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ema_alpha: float = 0.3,
    ):
        """
        初始化语义边界吸附器

        Args:
            semantic_threshold: 语义相似度阈值 (0-1)
                超过此值认为是同一话题，低于此值触发 Flush
            short_text_threshold: 短文本强吸附阈值（tokens）
                少于此值的文本强制吸附，防止错误切分
            embedding_model: Embedding 模型名称
            ema_alpha: 指数移动平均系数
                用于更新话题核心向量
        """
        if not 0 <= semantic_threshold <= 1:
            raise ValueError("semantic_threshold 必须在 [0, 1] 范围内")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha 必须在 (0, 1] 范围内")

        self.semantic_threshold = semantic_threshold
        self.short_text_threshold = short_text_threshold
        self.ema_alpha = ema_alpha

        # 初始化 Embedding 服务（延迟加载）
        self._embedding_service = None
        self._embedding_model_name = embedding_model

        # 文本到向量的映射（用于维护话题核心）
        self._anchor_cache: dict = {}

        logger.info(
            f"SemanticBoundaryAdsorber 初始化: "
            f"threshold={semantic_threshold}, "
            f"short_text={short_text_threshold}"
        )

    @property
    def embedding_service(self):
        """延迟加载 Embedding 服务"""
        if self._embedding_service is None:
            from hivememory.core.embedding import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def compute_similarity(
        self,
        anchor_text: str,
        topic_kernel: Optional[List[float]]
    ) -> float:
        """
        计算语义相似度

        如果 topic_kernel 为 None，返回 0（强制吸附）

        Args:
            anchor_text: 锚点文本（新 Block 的 User Query）
            topic_kernel: 话题核心向量

        Returns:
            float: 相似度 (0-1)
        """
        if not topic_kernel or not anchor_text:
            return 0.0

        try:
            # 将话题核心向量转换为文本表示（用于相似度计算）
            # 这里我们使用缓存中的文本，如果没有则使用第一个锚点
            similarity = self._compute_cosine_similarity(anchor_text, topic_kernel)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Embedding 计算失败: {e}")
            return 0.0

    def _compute_cosine_similarity(
        self,
        text: str,
        reference_vector: List[float]
    ) -> float:
        """计算文本与参考向量的余弦相似度"""
        import numpy as np

        text_vector = self.embedding_service.encode(text, normalize=True)
        ref_array = np.array(reference_vector)
        text_array = np.array(text_vector)

        return float(np.dot(ref_array, text_array))

    def should_adsorb(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer
    ) -> Tuple[bool, Optional[FlushReason]]:
        """
        判断是否吸附

        判定流程：
            1. 检查 Block 是否完整
            2. 短文本强吸附
            3. 语义相似度判定

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

        # 1. 短文本强吸附
        if new_block.total_tokens < self.short_text_threshold:
            logger.debug(
                f"短文本强吸附: {new_block.total_tokens} < {self.short_text_threshold}"
            )
            return True, FlushReason.SHORT_TEXT_ADSORB

        # 2. 语义相似度判定
        if buffer.topic_kernel_vector is not None:
            similarity = self.compute_similarity(
                new_block.anchor_text,
                buffer.topic_kernel_vector
            )

            logger.debug(
                f"语义相似度: {similarity:.3f} (阈值: {self.semantic_threshold})"
            )

            if similarity < self.semantic_threshold:
                logger.debug("语义漂移，触发 Flush")
                return False, FlushReason.SEMANTIC_DRIFT

        # 默认吸附
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
            new_vector = self.embedding_service.encode(new_block.anchor_text, normalize=True)

            # 缓存锚点文本
            cache_key = f"{buffer.buffer_id}:{new_block.block_id}"
            self._anchor_cache[cache_key] = {
                "text": new_block.anchor_text,
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
                    self.ema_alpha * new_vec + (1 - self.ema_alpha) * old_vec
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
]
