"""
HiveMemory - Generation 模块接口抽象层

定义了记忆生成模块的所有核心接口，遵循依赖倒置原则，便于扩展和测试。

接口列表:
- ValueGater: 价值评估器接口
- MemoryExtractor: 记忆提取器接口
- Deduplicator: 查重与演化器接口

作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

from hivememory.core.models import MemoryAtom, StreamMessage
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    MergeResult,
)


# ========== 接口定义 ==========

class BaseMemoryExtractor(ABC):
    """
    记忆提取器接口

    职责:
        调用 LLM 将自然对话转换为结构化的记忆草稿。

    实现策略:
        - LiteLLM 统一接口
        - Pydantic 输出解析
        - JSON 容错与重试
    """

    @abstractmethod
    def extract(
        self,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> Optional["ExtractedMemoryDraft"]:
        """
        提取记忆草稿

        Args:
            transcript: 格式化的对话文本
            metadata: 元信息 (session_id, user_id, agent_id, timestamp)

        Returns:
            ExtractedMemoryDraft: 提取的记忆草稿，失败时返回 None

        Raises:
            ExtractionError: LLM 调用失败或解析失败时抛出

        Examples:
            >>> extractor = LLMMemoryExtractor(llm_config)
            >>> draft = extractor.extract(
            ...     transcript="User: 如何解析日期?\nAssistant: 使用 datetime...",
            ...     metadata={"user_id": "user123", "session_id": "sess456"}
            ... )
            >>> print(draft.title)
            "Python 日期解析方法"
        """
        pass

    def merge(
        self,
        old_content: str,
        metadata: Dict[str, Any],
    ) -> Optional["MergeResult"]:
        """
        执行记忆合并 (Mode C: UPDATE)

        调用 LLM 将旧内容与修改指令合并，生成新内容和变更日志。
        默认实现返回 None (NoOp)，子类可覆盖。

        Args:
            old_content: 目标记忆的当前内容
            metadata: 元信息，包含:
                - instruction: 修改指令
                - new_content: 新素材 (可能为空)
                - memory_title: 目标记忆标题
                - memory_alias: 目标记忆别名
                - transcript: 近期对话上下文

        Returns:
            MergeResult: 合并结果，失败时返回 None
        """
        return None


class BaseDeduplicator(ABC):
    """
    查重与演化器接口

    职责:
        检测重复记忆，支持知识更新与合并。

    决策逻辑:
        - 相似度 > 0.95 + 内容一致 → TOUCH
        - 0.75 < 相似度 < 0.95 → UPDATE
        - 相似度 < 0.75 → CREATE
    """

    @abstractmethod
    def check_duplicate(
        self,
        draft: "ExtractedMemoryDraft",
        candidates: list,
    ) -> "Tuple[DuplicateDecision, Optional[MemoryAtom]]":
        """
        检查记忆草稿是否重复（纯决策，无 I/O）

        Args:
            draft: LLM 提取的记忆草稿
            candidates: 调用方预先检索的候选结果列表（空列表 → CREATE）

        Returns:
            (DuplicateDecision, Optional[MemoryAtom])
        """
        pass

    @abstractmethod
    def merge_memory(
        self,
        existing: MemoryAtom,
        new_draft: "ExtractedMemoryDraft"
    ) -> MemoryAtom:
        """
        合并现有记忆与新草稿（知识演化）

        策略:
            - 保留高置信度内容
            - 追加新信息到 payload
            - 合并标签集合
            - 记录版本历史

        Args:
            existing: 现有记忆原子
            new_draft: 新的记忆草稿

        Returns:
            MemoryAtom: 合并后的记忆原子

        Examples:
            >>> merged = dedup.merge_memory(existing, draft)
            >>> print(merged.payload.content)
            "旧内容...\n\n## 更新 (2025-12-23)\n新内容..."
        """
        pass


# ========== 导出列表 ==========

__all__ = [
    "BaseMemoryExtractor",
    "BaseDeduplicator",
]
