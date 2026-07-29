"""
HiveMemory - 查重与演化管理器 (Deduplicator)

职责:
    检测重复记忆，输出 CREATE / UPDATE / TOUCH / DISCARD 决策。

决策逻辑 (PROJECT.md 4.2 Step 3):
    - 相似度 > 0.95 + 内容一致 → TOUCH (仅更新访问时间)
    - 0.75 < 相似度 < 0.95 → UPDATE (知识演化)
    - 相似度 < 0.75 → CREATE (创建新记忆)

作者: HiveMemory Team
"""

import re
import logging
from typing import Optional, Tuple

from hivememory.system.config import DeduplicatorConfig
from hivememory.core.models import (
    MemoryAtom,
)
from hivememory.engines.generation.models import DuplicateDecision, ExtractedMemoryDraft
from hivememory.engines.generation.interfaces import BaseDeduplicator

logger = logging.getLogger(__name__)


class MemoryDeduplicator(BaseDeduplicator):
    """
    记忆查重与演化管理器

    工作流程:
        1. 向量检索 Top-1 最相似记忆
        2. 计算相似度分数
        3. 根据决策矩阵判断操作
        4. 将是否合并的决策交给 generation engine 执行

    决策矩阵:
        | 相似度范围      | 内容一致性  | 决策      | 操作           |
        |----------------|------------|----------|---------------|
        | > 0.95         | 是         | TOUCH    | 更新访问时间   |
        | > 0.95         | 否         | UPDATE   | 内容演化       |
        | 0.75 - 0.95    | -          | UPDATE   | 内容合并       |
        | < 0.75         | -          | CREATE   | 创建新记忆     |

    Examples:
        >>> dedup = MemoryDeduplicator(config)
        >>> decision, existing = dedup.check_duplicate(draft, candidates)
    """

    def __init__(
        self,
        config: DeduplicatorConfig,
    ):
        self.config = config

    def check_duplicate(
        self,
        draft: ExtractedMemoryDraft,
        candidates: list,
    ) -> Tuple[DuplicateDecision, Optional[MemoryAtom]]:
        """
        纯决策：在调用方传入的候选列表上执行查重逻辑，无 I/O。

        Args:
            draft: LLM 提取的记忆草稿
            candidates: 搜索结果列表（每项含 "score" 和 "memory"）

        Returns:
            (DuplicateDecision, Optional[MemoryAtom])
        """
        if not candidates:
            return DuplicateDecision.CREATE, None

        top_result = candidates[0]
        similarity_score = top_result["score"]
        existing_memory = top_result["memory"]

        logger.info(
            f"找到相似记忆: '{existing_memory.index.title}' "
            f"(相似度: {similarity_score:.3f})"
        )

        decision = self._make_decision(
            similarity_score=similarity_score,
            draft=draft,
            existing=existing_memory,
        )
        logger.info(f"查重决策: {decision.value}")
        return decision, existing_memory

    def _make_decision(
        self,
        similarity_score: float,
        draft: ExtractedMemoryDraft,
        existing: MemoryAtom
    ) -> DuplicateDecision:
        """
        根据相似度和内容一致性做出决策

        Args:
            similarity_score: 向量相似度分数
            draft: 新草稿
            existing: 现有记忆

        Returns:
            DuplicateDecision: 决策结果
        """
        # 情况 1: 高相似度 (> 0.95)
        if similarity_score > self.config.high_similarity_threshold:
            # 检查内容是否完全一致
            if self._is_content_identical(draft, existing):
                logger.debug("高相似度 + 内容一致 → TOUCH")
                return DuplicateDecision.TOUCH
            else:
                logger.debug("高相似度 + 内容不同 → UPDATE (微小演化)")
                return DuplicateDecision.UPDATE

        # 情况 2: 中等相似度 (0.75 - 0.95)
        elif similarity_score > self.config.low_similarity_threshold:
            logger.debug("中等相似度 → UPDATE (知识合并)")
            return DuplicateDecision.UPDATE

        # 情况 3: 低相似度 (< 0.75)
        else:
            logger.debug("低相似度 → CREATE (新记忆)")
            return DuplicateDecision.CREATE

    def _is_content_identical(
        self,
        draft: ExtractedMemoryDraft,
        existing: MemoryAtom
    ) -> bool:
        """
        判断内容是否完全一致

        策略:
            - 比较 title 精确匹配
            - 比较 content 字符级相似度 (> 90%)

        Args:
            draft: 新草稿
            existing: 现有记忆

        Returns:
            bool: 是否一致
        """
        # 标题完全一致
        if draft.title != existing.index.title:
            return False

        # 内容相似度
        draft_content = draft.content.strip()
        existing_content = existing.payload.content.strip()

        # 简单字符级相似度 (Jaccard)
        similarity = self._calculate_text_similarity(draft_content, existing_content)

        return similarity > self.config.content_similarity_threshold

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度 (简单 Jaccard 相似度)

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度 (0.0-1.0)
        """
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

class NoOpDeduplicator(BaseDeduplicator):
    """
    No-Op 查重器

    不执行查重操作，总是返回 CREATE (判定为新记忆)。
    用于在配置未启用查重器时作为默认实现。
    """

    def check_duplicate(
        self,
        draft: ExtractedMemoryDraft,
        candidates: list,
    ) -> Tuple[DuplicateDecision, Optional[MemoryAtom]]:
        return DuplicateDecision.CREATE, None


# 便捷函数
def create_deduplicator(
    config: DeduplicatorConfig,
) -> BaseDeduplicator:
    if not config.enabled:
        logger.info("Deduplicator 已禁用 (No-Op)")
        return NoOpDeduplicator()
    logger.info("Deduplicator 已启用")
    return MemoryDeduplicator(config=config)


__all__ = [
    "MemoryDeduplicator",
    "NoOpDeduplicator",
    "create_deduplicator",
]
