"""
HiveMemory - Generation 模块接口抽象层

定义了记忆生成模块的所有核心接口，遵循依赖倒置原则，便于扩展和测试。

接口列表:
- ValueGater: 价值评估器接口
- MemoryExtractor: 记忆提取器接口
- Deduplicator: 查重与演化器接口
- TriggerStrategy: 触发策略接口

作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

from hivememory.core.models import ConversationMessage, MemoryAtom


# ========== 枚举定义 ==========

class DuplicateDecision(str, Enum):
    """
    查重决策类型

    Attributes:
        CREATE: 创建新记忆
        UPDATE: 更新现有记忆（知识演化）
        TOUCH: 仅更新访问时间（完全重复）
        DISCARD: 丢弃（低质量重复）
    """
    CREATE = "create"
    UPDATE = "update"
    TOUCH = "touch"
    DISCARD = "discard"


# ========== 接口定义 ==========

class ValueGater(ABC):
    """
    价值评估器接口

    职责:
        判断对话片段是否有长期记忆价值，过滤无用的闲聊和噪音。

    实现策略:
        - 规则引擎: 基于关键词黑名单
        - LLM 辅助: 使用轻量级模型判断
        - 混合策略: 规则 + LLM 结合
    """

    @abstractmethod
    def evaluate(self, messages: List[ConversationMessage]) -> bool:
        """
        评估对话是否有价值

        Args:
            messages: 对话消息列表

        Returns:
            bool: True 表示有价值，False 表示无价值

        Examples:
            >>> gater = RuleBasedGater()
            >>> messages = [ConversationMessage(role="user", content="你好")]
            >>> gater.evaluate(messages)
            False
        """
        pass


class MemoryExtractor(ABC):
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


class Deduplicator(ABC):
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
        threshold: float = 0.75
    ) -> DuplicateDecision:
        """
        检查记忆草稿是否重复

        Args:
            draft: LLM 提取的记忆草稿
            threshold: 相似度阈值，默认 0.75

        Returns:
            DuplicateDecision: 决策结果 (CREATE/UPDATE/TOUCH/DISCARD)

        Examples:
            >>> dedup = MemoryDeduplicator(storage)
            >>> decision = dedup.check_duplicate(draft)
            >>> if decision == DuplicateDecision.UPDATE:
            ...     merged = dedup.merge_memory(existing, draft)
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

# ========== 辅助数据类 ==========

class ExtractedMemoryDraft:
    """
    提取的记忆草稿 - 用于 LLM 输出解析

    注意: 该类在 generation/extractor.py 中使用 Pydantic 重新定义，
    这里仅作为类型提示占位符。

    Attributes:
        title: 简洁标题
        summary: 一句话摘要
        tags: 语义标签列表
        memory_type: 记忆类型 (CODE_SNIPPET/FACT/...)
        content: Markdown 内容
        confidence_score: 置信度 (0.0-1.0)
        has_value: 是否有长期价值
    """
    pass


# ========== 异常定义 ==========

class ExtractionError(Exception):
    """记忆提取失败异常"""
    pass


class DeduplicationError(Exception):
    """查重处理失败异常"""
    pass


# ========== 导出列表 ==========

__all__ = [
    # 枚举
    "DuplicateDecision",
    "TriggerReason",

    # 接口
    "ValueGater",
    "MemoryExtractor",
    "Deduplicator",
    "TriggerStrategy",

    # 数据类
    "ExtractedMemoryDraft",

    # 异常
    "ExtractionError",
    "DeduplicationError",
]
