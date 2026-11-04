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
    RevisionResult,
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
    async def extract(self, transcript: str) -> list["ExtractedMemoryDraft"]:
        """
        Mode A：从一段话题材料中提取彼此独立的记忆草稿

        Args:
            transcript: 格式化的话题材料

        Returns:
            记忆草稿列表；模型输出为空或无法解析时为空列表

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或输出达到上限被截断
        """
        pass

    async def extract_write(
        self, transcript: str, *, content: str, reason: str | None, title: str | None
    ) -> Optional["ExtractedMemoryDraft"]:
        """
        Mode B：以 WRITE 提交的内容为核心，结合背景对话结构化为一条草稿

        默认实现返回 None (NoOp)，调用方据此使用 fallback 草稿。

        Args:
            transcript: 格式化的背景对话
            content: WRITE 提交的内容
            reason: WRITE 给出的理由，可能为空
            title: WRITE 拟定的标题，可能为空

        Returns:
            ExtractedMemoryDraft: 草稿；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽
        """
        return None

    async def revise(
        self,
        old_content: str,
        metadata: dict[str, Any],
    ) -> Optional["RevisionResult"]:
        """
        执行记忆修订 (Mode C: UPDATE)

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
            RevisionResult: 修订结果；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽
        """
        return None

    async def merge(self, retained: MemoryAtom, absorbed: MemoryAtom) -> MergeResult | None:
        """融合两条正式记忆，返回完整索引与正文；未启用模型时没有产出。"""
        return None


class BaseDeduplicator(ABC):
    """
    查重与演化器接口

    职责:
        检测重复记忆，命中相似内容时由生成引擎新建并提名合并。

    决策逻辑:
        - 高于上限且内容一致 → IDENTICAL
        - 高于上限但内容不同，或高于下限且不高于上限 → SIMILAR
        - 不高于下限 → DISTINCT
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
            candidates: 调用方预先检索的候选结果列表（空列表 → DISTINCT）

        Returns:
            (DuplicateDecision, Optional[MemoryAtom])
        """
        pass


# ========== 导出列表 ==========

__all__ = [
    "BaseMemoryExtractor",
    "BaseDeduplicator",
]
