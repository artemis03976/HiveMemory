"""Compatibility wrapper and shared constants for memory atom rendering."""

from __future__ import annotations

from enum import Enum

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler.handlers.memory_atom import (
    AGENT_PROFILE_ITEM_TEMPLATE,
    FULL_ITEM_TEMPLATE,
    INDEX_ITEM_TEMPLATE,
    _format_confidence,
    _render_agent_profile,
    _render_dense_embedding,
    _render_full_context,
    _render_index_context,
    _render_sparse_embedding,
    _truncate_content,
)


MEMORY_HEADER = """<memory_context>
[System Guidance]: 帕秋莉 (记忆库的管理者) 为你取回了以下相关的历史记忆与可用子代理。
你可以将记忆信息视为你脑海里自然而然浮现的"潜意识"，作为背景知识直接融合到你的思考中，无需刻意生硬地声明"根据记忆显示"。
"""

MEMORY_FOOTER = """
\n[System Guidance]:
- 若上述记忆摘要符合当前用户意图，但摘要信息不足，希望查看完整的记忆内容，请立即使用 `⟪ READ | alias | ⟫` 指令（**严禁自行猜测或编造**）。
- 带有 [未验证] 或 (警告：陈旧) 状态的记忆可能包含错误或过时信息，请结合常识注意甄别。
- 若任务需要专项能力（如数据分析、代码生成等），且上方列出了对应子代理，请优先使用 `⟪ CALL | agent_alias | topic="..." ⟫` 委托给子代理执行，不要自行承担。
</memory_context>
"""


class RenderFormat(str, Enum):
    """Rendering format enum retained for backward compatibility."""

    XML = "xml"
    MARKDOWN = "markdown"


class MemoryAtomRenderer:
    """Deprecated compatibility facade; use MemoryCompiler instead."""

    @staticmethod
    def for_dense_embedding(memory: MemoryAtom) -> str:
        return _render_dense_embedding(memory)

    @staticmethod
    def for_sparse_embedding(memory: MemoryAtom) -> str:
        return _render_sparse_embedding(memory)

    @staticmethod
    def for_full_context(
        memory: MemoryAtom,
        max_content_length: int = 500,
        stale_days: int = 90,
    ) -> str:
        return _render_full_context(
            memory=memory,
            max_content_length=max_content_length,
            stale_days=stale_days,
        )

    @staticmethod
    def for_index_context(
        memory: MemoryAtom,
        max_summary_length: int = 100,
        stale_days: int = 90,
    ) -> str:
        return _render_index_context(
            memory=memory,
            max_summary_length=max_summary_length,
            stale_days=stale_days,
        )

    @staticmethod
    def _format_confidence(memory: MemoryAtom) -> str:
        return _format_confidence(memory)

    @staticmethod
    def _truncate_content(content: str, max_length: int) -> str:
        return _truncate_content(content, max_length)

    @staticmethod
    def for_agent_profile(memory: MemoryAtom) -> str:
        return _render_agent_profile(memory)


__all__ = [
    "MemoryAtomRenderer",
    "RenderFormat",
    "MEMORY_HEADER",
    "MEMORY_FOOTER",
    "FULL_ITEM_TEMPLATE",
    "INDEX_ITEM_TEMPLATE",
    "AGENT_PROFILE_ITEM_TEMPLATE",
]
