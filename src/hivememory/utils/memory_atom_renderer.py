"""
MemoryAtom 通用渲染器

职责:
    将 MemoryAtom 渲染为不同用途的文本格式:
    - for_dense_embedding: 用于稠密向量生成的文本
    - for_sparse_embedding: 用于稀疏向量生成的文本
    - for_full_context: 用于注入 LLM 上下文的完整记忆渲染
    - for_index_context: 用于注入 LLM 上下文的索引/摘要渲染

设计原则:
    - 单一职责: 仅负责渲染逻辑，不处理业务逻辑
    - 无状态: 所有方法都是静态方法
    - 模板集中: 所有 LLM 上下文注入模板统一在此管理
    - 可扩展: 未来可添加更多渲染场景 (UI展示、调试输出等)
"""

from typing import Optional
from enum import Enum

from hivememory.core.models import MemoryAtom, VerificationStatus
from hivememory.utils.time_formatter import TimeFormatter, Language


# ==========================================
# 统一的 Header 与 Footer
# ==========================================

MEMORY_HEADER = """<memory_context>
[System Guidance]: 帕秋莉 (记忆库的管理者) 为你取回了以下相关的历史记忆与可用工具。
你可以将这些信息视为你脑海里自然而然浮现的"潜意识"，作为背景知识直接融合到你的思考中，无需刻意生硬地声明"根据记忆显示"。
"""

MEMORY_FOOTER = """
\n[System Guidance]:
- 若上述记忆摘要符合当前用户意图，但摘要信息不足，希望查看完整的记忆内容，请立即使用 `⟪ READ | alias | ⟫` 指令（**严谨自行猜测或编造**）。
- 带有 [未验证] 或 (警告：陈旧) 状态的记忆可能包含错误或过时信息，请结合常识注意甄别。
</memory_context>
"""

# ==========================================
# Full View 模板 (用于策略 A/B: 全量加载/瀑布流)
# ==========================================

FULL_ITEM_TEMPLATE = """
<memory alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}

[完整内容]:
{content}
{history}
</memory>"""

# ==========================================
# Compact/Index View 模板 (用于策略 B/C: 瀑布流/懒加载)
# ==========================================

INDEX_ITEM_TEMPLATE = """
<memory_index alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}
- **内容摘要**: {summary}
</memory_index>"""


class RenderFormat(str, Enum):
    """渲染格式枚举 (保留用于向后兼容)"""
    XML = "xml"
    MARKDOWN = "markdown"


class MemoryAtomRenderer:
    """
    MemoryAtom 通用渲染器

    将记忆原子渲染为不同用途的文本格式，集中管理所有渲染逻辑。
    """

    # ========== Embedding 渲染 ==========

    @staticmethod
    def for_dense_embedding(memory: MemoryAtom) -> str:
        """
        渲染用于稠密向量 (Dense Embedding) 生成的文本

        格式: Title: {title}\nType: {type}\nTags: {tags}\nSummary: {summary}

        Args:
            memory: 记忆原子

        Returns:
            用于 dense embedding 的文本

        Examples:
            >>> from hivememory.core.models import MemoryAtom, IndexLayer, MemoryType
            >>> index = IndexLayer(
            ...     title="Python parse_date 函数",
            ...     summary="基于 datetime 库的日期解析工具",
            ...     tags=["python", "datetime"],
            ...     memory_type=MemoryType.CODE_SNIPPET
            ... )
            >>> MemoryAtomRenderer.for_dense_embedding(index)
            'Title: Python parse_date 函数\\nType: code_snippet\\nTags: python, datetime\\nSummary: 基于 datetime 库的日期解析工具'
        """
        return (
            f"Title: {memory.index.title}\n"
            f"Type: {memory.index.memory_type.value}\n"
            f"Tags: {', '.join(memory.index.tags)}\n"
            f"Summary: {memory.index.summary}"
        )

    @staticmethod
    def for_sparse_embedding(memory: MemoryAtom) -> str:
        """
        渲染用于稀疏向量 (Sparse Embedding) 生成的文本

        格式: "{title} {title} {tags_string} {tags_string} {summary}"

        Title 和 tags 重复出现以增加其在稀疏向量中的权重。
        这用于 BGE-M3 的稀疏向量生成，捕获精准实体匹配。

        Args:
            memory: 记忆原子

        Returns:
            用于 sparse embedding 的文本

        Examples:
            >>> from hivememory.core.models import IndexLayer, MemoryType
            >>> index = IndexLayer(
            ...     title="Python parse_date 函数",
            ...     summary="基于 datetime 库的日期解析工具",
            ...     tags=["python", "datetime", "utils"],
            ...     memory_type=MemoryType.CODE_SNIPPET
            ... )
            >>> MemoryAtomRenderer.for_sparse_embedding(index)
            'Python parse_date 函数 Python parse_date 函数 python datetime utils python datetime utils 基于 datetime 库的日期解析工具'
        """
        tags_string = " ".join(memory.index.tags)
        return (
            f"{memory.index.title} {memory.index.title} "
            f"{tags_string} {tags_string} "
            f"{memory.index.summary}"
        )

    # ========== LLM 上下文渲染 ==========

    @staticmethod
    def for_full_context(
        memory: MemoryAtom,
        max_content_length: int = 500,
        stale_days: int = 90,
    ) -> str:
        """
        渲染用于注入 LLM 上下文的完整记忆文本

        使用统一的 FULL_ITEM_TEMPLATE (XML 结构 + MD 内容)。

        Args:
            memory: 记忆原子
            max_content_length: 内容最大长度
            stale_days: 记忆被视为陈旧的天数

        Returns:
            渲染后的单条记忆文本
        """
        content = MemoryAtomRenderer._truncate_content(memory.payload.content, max_content_length)
        confidence_str = MemoryAtomRenderer._format_confidence(memory)
        alias = memory.get_alias()
        tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
        time_str = TimeFormatter(language=Language.CHINESE, stale_days=stale_days).format(memory.meta.updated_at)

        # 构建版本历史
        history = ""
        if memory.payload.history_summary:
            history_lines = ["\n**Change Log:**"]
            history_lines.extend([f"- {item}" for item in memory.payload.history_summary])
            history = "\n".join(history_lines)

        return FULL_ITEM_TEMPLATE.format(
            alias=alias,
            title=memory.index.title,
            type=memory.index.memory_type.value,
            time=time_str,
            confidence=confidence_str,
            tags=tags,
            content=content,
            history=history,
        )

    @staticmethod
    def for_index_context(
        memory: MemoryAtom,
        max_summary_length: int = 100,
        stale_days: int = 90,
    ) -> str:
        """
        渲染用于注入 LLM 上下文的索引/摘要文本

        使用统一的 INDEX_ITEM_TEMPLATE (XML 结构 + MD 内容)。
        适用于瀑布流降级和懒加载场景。

        Args:
            memory: 记忆原子
            max_summary_length: 摘要最大长度
            stale_days: 记忆被视为陈旧的天数

        Returns:
            渲染后的索引文本
        """
        alias = memory.get_alias()
        confidence_str = MemoryAtomRenderer._format_confidence(memory)
        tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
        time_str = TimeFormatter(language=Language.CHINESE, stale_days=stale_days).format(memory.meta.updated_at)

        summary = memory.index.summary
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length] + "..."

        return INDEX_ITEM_TEMPLATE.format(
            alias=alias,
            title=memory.index.title,
            type=memory.index.memory_type.value,
            time=time_str,
            confidence=confidence_str,
            tags=tags,
            summary=summary,
        )

    # ========== 内部辅助方法 ==========

    @staticmethod
    def _format_confidence(memory: MemoryAtom) -> str:
        """
        格式化置信度字符串

        Args:
            memory: 记忆原子

        Returns:
            格式化后的置信度字符串
        """
        score = memory.meta.confidence_score
        status = memory.meta.verification_status

        # 验证状态标记
        status_str = ""
        if status == VerificationStatus.VERIFIED:
            status_str = " [已验证]"
        elif status == VerificationStatus.DEPRECATED:
            status_str = " [已废弃]"
        elif status == VerificationStatus.HALLUCINATION:
            status_str = " [警告：幻觉]"
        elif score < 0.7:
            status_str = " [未验证]"

        # 分数格式化
        if score >= 0.9:
            return f"{score:.0%} (高){status_str}"
        elif score >= 0.7:
            return f"{score:.0%} (中){status_str}"
        else:
            return f"{score:.0%} (低){status_str}"

    @staticmethod
    def _truncate_content(content: str, max_length: int) -> str:
        """
        智能截断过长的内容

        尝试在句子边界截断，而非生硬切断。

        Args:
            content: 原始内容
            max_length: 最大长度

        Returns:
            截断后的内容
        """
        if len(content) <= max_length:
            return content

        # 智能截断：尝试在句子边界截断
        truncated = content[:max_length]

        # 尝试找到最后一个完整句子
        for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
            last_sep = truncated.rfind(sep)
            if last_sep > max_length // 2:
                truncated = truncated[:last_sep + len(sep)]
                break

        return truncated + "\n\n[...部分内容已截断，如需阅读完整内容请使用 READ 指令读取...]"


__all__ = [
    "MemoryAtomRenderer",
    "RenderFormat",
    "MEMORY_HEADER",
    "MEMORY_FOOTER",
    "FULL_ITEM_TEMPLATE",
    "INDEX_ITEM_TEMPLATE",
]
