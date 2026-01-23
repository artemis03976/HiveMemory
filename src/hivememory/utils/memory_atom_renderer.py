"""
MemoryAtom 通用渲染器

职责:
    将 MemoryAtom 渲染为不同用途的文本格式:
    - for_dense_embedding: 用于稠密向量生成的文本
    - for_sparse_embedding: 用于稀疏向量生成的文本
    - for_llm_context: 用于注入 LLM 上下文的自然语言文本

设计原则:
    - 单一职责: 仅负责渲染逻辑，不处理业务逻辑
    - 无状态: 所有方法都是静态方法
    - 可扩展: 未来可添加更多渲染场景 (UI展示、调试输出等)
"""

from typing import Literal, Optional
from enum import Enum

from hivememory.core.models import MemoryAtom, VerificationStatus


class RenderFormat(str, Enum):
    """渲染格式枚举"""
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
    def for_llm_context(
        memory: MemoryAtom,
        format: Literal["xml", "markdown"] = "xml",
        index: Optional[int] = None,
        max_content_length: int = 500,
        show_artifacts: bool = False,
        formatted_time: str = "",
    ) -> str:
        """
        渲染用于注入 LLM 上下文的自然语言文本

        Args:
            memory: 记忆原子
            format: 输出格式，"xml" 或 "markdown"
            index: 索引编号 (XML 格式需要，Markdown 为 None)
            max_content_length: 内容最大长度
            show_artifacts: 是否显示原始数据链接
            formatted_time: 已格式化的时间字符串 (由 TimeFormatter 生成)

        Returns:
            渲染后的单条记忆文本
        """
        # XML 模板
        XML_TEMPLATE = """
<memory_block id="{id}" type="{type}">
    [标签]: {tags}
    (时间): {time}
    [置信度]: {confidence}
    [内容]:
    {content}
</memory_block>"""

        # Markdown 模板
        MD_TEMPLATE = """
### 📌 {title}

- **类型**: `{type}`
- **标签**: {tags}
- **时间**: {time}
- **置信度**: {confidence}

{content}
{history}
{source}

---"""

        content = MemoryAtomRenderer._truncate_content(memory.payload.content, max_content_length)
        confidence_str = MemoryAtomRenderer._format_confidence(memory)

        if index is not None:  # XML 格式
            tags = ", ".join(f"#{tag}" for tag in memory.index.tags)
            tags_empty = "(无标签)"

            return XML_TEMPLATE.format(
                id=index,
                type=memory.index.memory_type.value,
                tags=tags if tags else tags_empty,
                time=formatted_time,
                confidence=confidence_str,
                content=content
            )
        else:  # Markdown 格式
            tags = ", ".join(f"`{tag}`" for tag in memory.index.tags)
            tags_empty = "(无标签)"

            # 构建版本历史
            history = ""
            if memory.payload.history_summary:
                history_lines = ["", "**Change Log:**"]
                history_lines.extend([f"- {item}" for item in memory.payload.history_summary])
                history = "\n".join(history_lines)

            # 构建原始数据引用
            source = ""
            if show_artifacts and memory.payload.artifacts.raw_source_url:
                source = f"\n\n**Source**: {memory.payload.artifacts.raw_source_url}"

            return MD_TEMPLATE.format(
                title=memory.index.title,
                type=memory.index.memory_type.value,
                tags=tags if tags else tags_empty,
                time=formatted_time,
                confidence=confidence_str,
                content=content,
                history=history,
                source=source
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
            return f"✓ {score:.0%} (高){status_str}"
        elif score >= 0.7:
            return f"~ {score:.0%} (中){status_str}"
        else:
            return f"? {score:.0%} (低-需验证){status_str}"

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

        return truncated + "\n\n[内容已截断。如需完整内容请询问。]"


__all__ = [
    "MemoryAtomRenderer",
    "RenderFormat",
]
