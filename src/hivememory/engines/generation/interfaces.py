"""
HiveMemory - Generation 模块接口抽象层

定义了记忆生成模块的所有核心接口，遵循依赖倒置原则，便于扩展和测试。

接口列表:
- ValueGater: 价值评估器接口
- MemoryExtractor: 记忆提取器接口
- Deduplicator: 查重与演化器接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from hivememory.core.models import MemoryAtom
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
        - 经 LLM 服务接口异步调用，等待模型期间不阻塞事件循环
        - Pydantic 输出解析
        - JSON 容错与提供商失败重试
    """

    @abstractmethod
    async def extract(
        self, transcript: str, metadata: dict[str, Any]
    ) -> Optional["ExtractedMemoryDraft"]:
        """
        提取记忆草稿

        Args:
            transcript: 格式化的对话文本
            metadata: 模式相关的生成元信息 (mode / write_content 等)；
                身份归属一律由 WorkspaceIdentity 承载，不在此传递

        Returns:
            ExtractedMemoryDraft: 提取的记忆草稿；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽

        Examples:
            >>> extractor = LLMMemoryExtractor(config, llm_service, llm_params)
            >>> draft = await extractor.extract(
            ...     transcript="User: 如何解析日期?\nAssistant: 使用 datetime...",
            ...     metadata={"mode": "B", "write_content": "待写入的记忆内容"}
            ... )
            >>> print(draft.title)
            "Python 日期解析方法"
        """
        pass

    async def merge(
        self,
        old_content: str,
        metadata: dict[str, Any],
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
            MergeResult: 合并结果；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽
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
    ) -> "tuple[DuplicateDecision, MemoryAtom | None]":
        """
        检查记忆草稿是否重复（纯决策，无 I/O）

        Args:
            draft: LLM 提取的记忆草稿
            candidates: 调用方预先检索的候选结果列表（空列表 → CREATE）

        Returns:
            (DuplicateDecision, Optional[MemoryAtom])
        """
        pass


# ========== 导出列表 ==========

__all__ = [
    "BaseMemoryExtractor",
    "BaseDeduplicator",
]
