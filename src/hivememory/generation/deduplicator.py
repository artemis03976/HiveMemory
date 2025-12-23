"""
HiveMemory - 查重与演化管理器 (Deduplicator)

职责:
    检测重复记忆，支持知识更新与合并。

决策逻辑 (PROJECT.md 4.2 Step 3):
    - 相似度 > 0.95 + 内容一致 → TOUCH (仅更新访问时间)
    - 0.75 < 相似度 < 0.95 → UPDATE (知识演化)
    - 相似度 < 0.75 → CREATE (创建新记忆)

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
from typing import Optional, Tuple
from datetime import datetime
from uuid import UUID

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.generation.interfaces import Deduplicator, DuplicateDecision
from hivememory.generation.extractor import ExtractedMemoryDraft

logger = logging.getLogger(__name__)


class MemoryDeduplicator(Deduplicator):
    """
    记忆查重与演化管理器

    工作流程:
        1. 向量检索 Top-1 最相似记忆
        2. 计算相似度分数
        3. 根据决策矩阵判断操作
        4. 如需合并，执行知识演化策略

    决策矩阵:
        | 相似度范围      | 内容一致性  | 决策      | 操作           |
        |----------------|------------|----------|---------------|
        | > 0.95         | 是         | TOUCH    | 更新访问时间   |
        | > 0.95         | 否         | UPDATE   | 内容演化       |
        | 0.75 - 0.95    | -          | UPDATE   | 内容合并       |
        | < 0.75         | -          | CREATE   | 创建新记忆     |

    Examples:
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> dedup = MemoryDeduplicator(QdrantMemoryStore())
        >>> decision, existing = dedup.check_duplicate(draft)
        >>> if decision == DuplicateDecision.UPDATE:
        ...     merged = dedup.merge_memory(existing, draft)
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        high_similarity_threshold: float = 0.95,
        low_similarity_threshold: float = 0.75,
        content_similarity_threshold: float = 0.9,
    ):
        """
        初始化查重管理器

        Args:
            storage: 向量存储实例 (QdrantMemoryStore)
            high_similarity_threshold: 高相似度阈值，默认 0.95
            low_similarity_threshold: 低相似度阈值，默认 0.75
            content_similarity_threshold: 内容相似度阈值，默认 0.9
        """
        self.storage = storage
        self.high_threshold = high_similarity_threshold
        self.low_threshold = low_similarity_threshold
        self.content_threshold = content_similarity_threshold

    def check_duplicate(
        self,
        draft: ExtractedMemoryDraft,
        threshold: float = 0.75
    ) -> Tuple[DuplicateDecision, Optional[MemoryAtom]]:
        """
        检查记忆草稿是否重复

        Args:
            draft: LLM 提取的记忆草稿
            threshold: 相似度阈值 (覆盖默认值)

        Returns:
            Tuple[DuplicateDecision, Optional[MemoryAtom]]:
                - 决策结果 (CREATE/UPDATE/TOUCH/DISCARD)
                - 现有记忆 (如果存在)

        Examples:
            >>> decision, existing = dedup.check_duplicate(draft)
            >>> if decision == DuplicateDecision.CREATE:
            ...     print("创建新记忆")
            >>> elif decision == DuplicateDecision.UPDATE:
            ...     merged = dedup.merge_memory(existing, draft)
        """
        try:
            # Step 1: 向量检索最相似记忆
            logger.debug(f"检索与 '{draft.title}' 相似的记忆...")

            # 使用 summary 作为查询文本（比 title 更有区分度）
            query_text = f"{draft.title} {draft.summary}"

            results = self.storage.search_memories(
                query_text=query_text,
                top_k=1,  # 只需要最相似的一条
                score_threshold=threshold,
            )

            # Step 2: 如果没有相似记忆，直接创建
            if not results:
                logger.debug("未找到相似记忆，判定为 CREATE")
                return DuplicateDecision.CREATE, None

            # Step 3: 提取最相似记忆及其分数
            top_result = results[0]
            similarity_score = top_result["score"]
            existing_memory = top_result["memory"]

            logger.info(
                f"找到相似记忆: '{existing_memory.index.title}' "
                f"(相似度: {similarity_score:.3f})"
            )

            # Step 4: 决策矩阵判断
            decision = self._make_decision(
                similarity_score=similarity_score,
                draft=draft,
                existing=existing_memory
            )

            logger.info(f"查重决策: {decision.value}")
            return decision, existing_memory

        except Exception as e:
            logger.error(f"查重检测失败: {e}", exc_info=True)
            # 失败时默认创建新记忆（安全策略）
            return DuplicateDecision.CREATE, None

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
        if similarity_score > self.high_threshold:
            # 检查内容是否完全一致
            if self._is_content_identical(draft, existing):
                logger.debug("高相似度 + 内容一致 → TOUCH")
                return DuplicateDecision.TOUCH
            else:
                logger.debug("高相似度 + 内容不同 → UPDATE (微小演化)")
                return DuplicateDecision.UPDATE

        # 情况 2: 中等相似度 (0.75 - 0.95)
        elif similarity_score > self.low_threshold:
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

        return similarity > self.content_threshold

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度 (简单 Jaccard 相似度)

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度 (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0

        # 转换为字符集
        set1 = set(text1)
        set2 = set(text2)

        # Jaccard 相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def merge_memory(
        self,
        existing: MemoryAtom,
        new_draft: ExtractedMemoryDraft
    ) -> MemoryAtom:
        """
        合并现有记忆与新草稿（知识演化）

        合并策略:
            1. 标题: 保留现有标题（更简洁）
            2. 摘要: 合并两者，取最全面的
            3. 标签: 合并标签集合（去重）
            4. 内容: 追加更新部分，记录版本
            5. 置信度: 取两者平均值
            6. 元信息: 更新时间戳，保留创建时间

        Args:
            existing: 现有记忆原子
            new_draft: 新的记忆草稿

        Returns:
            MemoryAtom: 合并后的记忆原子

        Examples:
            >>> merged = dedup.merge_memory(existing, draft)
            >>> print(merged.payload.content)
            # 包含旧内容和新内容，带时间戳标记
        """
        logger.info(f"合并记忆: '{existing.index.title}'")

        # 合并标签
        merged_tags = list(set(existing.index.tags) | set(new_draft.tags))
        if len(merged_tags) > 5:  # 限制标签数量
            merged_tags = merged_tags[:5]

        # 合并摘要 (选择更长的)
        merged_summary = (
            new_draft.summary
            if len(new_draft.summary) > len(existing.index.summary)
            else existing.index.summary
        )

        # 合并内容 (追加模式，带版本标记)
        merged_content = self._merge_content(
            old_content=existing.payload.content,
            new_content=new_draft.content
        )

        # 计算新置信度 (加权平均)
        merged_confidence = (
            existing.meta.confidence_score * 0.6 + new_draft.confidence_score * 0.4
        )

        # 构建合并后的 MemoryAtom
        merged_memory = MemoryAtom(
            id=existing.id,  # 保留原 ID
            meta=MetaData(
                source_agent_id=existing.meta.source_agent_id,
                user_id=existing.meta.user_id,
                session_id=existing.meta.session_id,
                created_at=existing.meta.created_at,  # 保留创建时间
                updated_at=datetime.utcnow(),  # 更新修改时间
                confidence_score=merged_confidence,
                access_count=existing.meta.access_count,  # 保留访问计数
                vitality_score=existing.meta.vitality_score,  # 保留生命力
            ),
            index=IndexLayer(
                title=existing.index.title,  # 保留原标题
                summary=merged_summary,
                tags=merged_tags,
                memory_type=existing.index.memory_type,  # 保留类型
                embedding=existing.index.embedding,  # 保留原向量（稍后重新生成）
            ),
            payload=PayloadLayer(
                content=merged_content,
            ),
            relations=existing.relations,  # 保留关系图
        )

        return merged_memory

    def _merge_content(self, old_content: str, new_content: str) -> str:
        """
        合并内容 (追加模式)

        策略:
            如果新内容与旧内容显著不同，则追加更新部分。

        Args:
            old_content: 旧内容
            new_content: 新内容

        Returns:
            str: 合并后的内容

        Examples:
            >>> content = dedup._merge_content("旧代码", "新代码")
            >>> print(content)
            旧代码

            ## 更新 (2025-12-23)
            新代码
        """
        # 如果内容高度相似，仅返回新内容
        similarity = self._calculate_text_similarity(old_content, new_content)
        if similarity > 0.9:
            return new_content

        # 追加模式
        timestamp = datetime.now().strftime("%Y-%m-%d")
        merged = f"{old_content}\n\n## 更新 ({timestamp})\n{new_content}"

        return merged


# 便捷函数
def create_default_deduplicator(storage) -> Deduplicator:
    """
    创建默认查重器

    Args:
        storage: QdrantMemoryStore 实例

    Returns:
        Deduplicator: 查重器实例
    """
    return MemoryDeduplicator(storage)


__all__ = [
    "MemoryDeduplicator",
    "create_default_deduplicator",
]
