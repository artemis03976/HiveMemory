"""
上下文渲染模块

职责:
    将检索到的记忆原子渲染为适合注入 LLM Context 的格式

渲染器类型:
    - FullContextRenderer: 完整渲染，超过字符上限则截断
    - CascadeContextRenderer: 瀑布式渲染，Top-N 完整 + 其余 Index
    - CompactContextRenderer: 仅渲染 Index 层信息

对应设计文档: PROJECT.md 5.2 节
"""

from typing import List, Optional, Tuple, Union
import logging

from hivememory.patchouli.config import FullRendererConfig, CascadeRendererConfig, CompactRendererConfig
from hivememory.core.models import MemoryAtom, estimate_tokens
from hivememory.engines.retrieval.models import RenderFormat
from hivememory.engines.retrieval.interfaces import BaseContextRenderer
from hivememory.utils import TimeFormatter, Language, MemoryAtomRenderer

logger = logging.getLogger(__name__)


# ========== 共享模板 ==========

# XML 模板
XML_HEADER = """<system_memory_context>
以下是从历史交互中检索到的相关记忆。
使用这些记忆来保持一致性并复用已有知识。
请注意 [标签] 和 (时间)。
"""

XML_FOOTER = """
</system_memory_context>

<instruction>
以上是你的记忆。如果某条记忆标记为 (Warning: Old) 或 [Unverified]，请在使用前验证。
如果需要更多关于某条记忆的详细信息，请向用户询问。
</instruction>"""

XML_FOOTER_SIMPLE = """
</system_memory_context>
"""

# Markdown 模板
MD_HEADER = """## 相关记忆上下文

以下是与当前对话相关的历史记忆，可用于保持一致性和复用知识：

---
"""

MD_FOOTER = """
---

> 如果某条记忆标记为 (Warning: Old) 或 [Unverified]，请在使用前验证。
"""

MD_FOOTER_SIMPLE = """
---
"""

# Index 视图模板
INDEX_XML_TEMPLATE = """
<memory_ref id="{id}" type="{type}">
    [标签]: {tags}
    [摘要]: {summary}
    [提示]: {hint}
</memory_ref>"""

INDEX_MD_TEMPLATE = """
###  {title} (摘要)

- **类型**: `{type}`
- **标签**: {tags}
- **摘要**: {summary}

> {hint}

---"""


# ========== 辅助函数 ==========

def _extract_memories(results: List) -> List[MemoryAtom]:
    """从结果列表中提取 MemoryAtom"""
    memories = []
    for item in results:
        if hasattr(item, 'memory'):
            memories.append(item.memory)
        elif isinstance(item, MemoryAtom):
            memories.append(item)
        else:
            logger.warning(f"未知的结果类型: {type(item)}")
    return memories


class FullContextRenderer(BaseContextRenderer):
    """
    完整上下文渲染器

    渲染所有 MemoryAtom 的完整内容，超过字符上限则直接截断。
    """
    def __init__(self, config: FullRendererConfig):
        """
        初始化渲染器

        Args:
            config: 完整渲染器配置
        """
        self.config = config

        # 解析渲染格式
        if isinstance(config.render_format, str):
            self.render_format = RenderFormat.XML if config.render_format.lower() == "xml" else RenderFormat.MARKDOWN
        else:
            self.render_format = config.render_format

        self.max_tokens = config.max_tokens
        self.max_content_length = config.max_content_length
        self.show_artifacts = config.show_artifacts
        self._time_formatter = TimeFormatter(language=Language.CHINESE, stale_days=config.stale_days)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        """
        渲染记忆列表为上下文字符串

        Args:
            results: SearchResult 列表或 MemoryAtom 列表
            render_format: 输出格式（可选，覆盖默认）

        Returns:
            渲染后的上下文字符串
        """
        fmt = render_format or self.render_format

        if not results:
            return ""

        memories = _extract_memories(results)
        if not memories:
            return ""

        if fmt == RenderFormat.XML:
            return self._render_xml(memories)
        else:
            return self._render_markdown(memories)

    def _render_format(self, memories: List[MemoryAtom], header: str, footer: str, use_index: bool) -> str:
        """
        通用渲染函数

        Args:
            memories: 记忆列表
            header: 头部模板
            footer: 尾部模板
            use_index: 是否使用索引编号（XML格式需要）
        """
        blocks = [header]
        total_length = len(header) + len(footer)

        for i, memory in enumerate(memories, 1):
            block = self._render_memory(memory, i if use_index else None)

            if total_length + len(block) > self.max_tokens:
                logger.debug(f"达到长度限制，截断至 {i-1} 条记忆")
                break

            blocks.append(block)
            total_length += len(block)

        blocks.append(footer)
        return "".join(blocks)

    def _render_xml(self, memories: List[MemoryAtom]) -> str:
        """渲染为 XML 格式"""
        return self._render_format(memories, XML_HEADER, XML_FOOTER, use_index=True)

    def _render_markdown(self, memories: List[MemoryAtom]) -> str:
        """渲染为 Markdown 格式"""
        return self._render_format(memories, MD_HEADER, MD_FOOTER, use_index=False)

    def _render_memory(self, memory: MemoryAtom, index: Optional[int] = None) -> str:
        """
        渲染单条记忆

        Args:
            memory: 记忆原子
            index: 索引编号（XML格式需要，Markdown为None）
        """
        time_str = self._time_formatter.format(memory.meta.updated_at)
        format_type = "xml" if index is not None else "markdown"

        return MemoryAtomRenderer.for_llm_context(
            memory=memory,
            format=format_type,
            index=index,
            max_content_length=self.max_content_length,
            show_artifacts=self.show_artifacts,
            formatted_time=time_str,
        )


class CascadeContextRenderer(BaseContextRenderer):
    """
    瀑布式上下文渲染器

    依次完整渲染 MemoryAtom，直到 Token 预算紧张时降级为 Index 层信息:
    1. Top-N 记忆强制完整渲染 (Payload)
    2. 其余按预算瀑布式降级为 Index 视图 (摘要+标签)
    3. 预算耗尽时停止渲染

    适用于需要平衡完整性和 Token 预算的场景。
    """

    def __init__(self, config: CascadeRendererConfig):
        """
        初始化瀑布式渲染器

        Args:
            config: 瀑布式渲染器配置
        """
        self.config = config
        self._time_formatter = TimeFormatter(language=Language.CHINESE, stale_days=90)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        """
        渲染记忆列表

        算法:
        1. Top-N (full_payload_count) 强制完整渲染
        2. 其余按预算瀑布式降级:
           - 预算充足 -> 完整 Payload
           - 预算紧张 -> Index Only (摘要+标签)
           - 预算耗尽 -> 停止渲染

        Args:
            results: SearchResult 列表或 MemoryAtom 列表
            render_format: 输出格式（可选，覆盖默认）

        Returns:
            渲染后的上下文字符串
        """
        if not results:
            return ""

        fmt = render_format
        if fmt is None:
            fmt = RenderFormat.XML if self.config.render_format.lower() == "xml" else RenderFormat.MARKDOWN

        memories = _extract_memories(results)
        if not memories:
            return ""

        if fmt == RenderFormat.XML:
            header, footer = XML_HEADER, XML_FOOTER_SIMPLE
        else:
            header, footer = MD_HEADER, MD_FOOTER_SIMPLE

        header_footer_tokens = estimate_tokens(header) + estimate_tokens(footer)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        rendered_blocks, _ = self._render_with_budget(memories, available_budget, fmt)

        if not rendered_blocks:
            return ""

        return header + "".join(rendered_blocks) + footer

    def _render_with_budget(
        self,
        memories: List[MemoryAtom],
        budget: int,
        fmt: RenderFormat
    ) -> Tuple[List[str], int]:
        """
        带预算的瀑布式渲染

        Args:
            memories: 记忆列表
            budget: Token 预算
            fmt: 渲染格式

        Returns:
            (rendered_blocks, remaining_budget)
        """
        rendered_blocks = []
        remaining_budget = budget

        for i, memory in enumerate(memories):
            index = i + 1

            # Top-N 强制完整渲染
            if i < self.config.full_payload_count:
                full_block = self._render_full_payload(memory, index, fmt)
                full_tokens = estimate_tokens(full_block)

                if full_tokens <= remaining_budget:
                    rendered_blocks.append(full_block)
                    remaining_budget -= full_tokens
                    continue
                # 预算不足，降级为 Index

            # 尝试 Index 视图渲染
            index_block = self._render_index_only(memory, index, fmt)
            index_tokens = estimate_tokens(index_block)

            if index_tokens <= remaining_budget:
                rendered_blocks.append(index_block)
                remaining_budget -= index_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(rendered_blocks)} 条)")
                break

        return rendered_blocks, remaining_budget

    def _render_full_payload(self, memory: MemoryAtom, index: int, fmt: RenderFormat) -> str:
        """
        渲染完整 Payload

        Args:
            memory: 记忆原子
            index: 索引编号
            fmt: 渲染格式

        Returns:
            渲染后的文本
        """
        time_str = self._time_formatter.format(memory.meta.updated_at)
        max_content_length = self.config.max_content_length

        if fmt == RenderFormat.XML:
            return MemoryAtomRenderer.for_llm_context(
                memory=memory,
                format="xml",
                index=index,
                max_content_length=max_content_length,
                show_artifacts=False,
                formatted_time=time_str,
            )
        else:
            return MemoryAtomRenderer.for_llm_context(
                memory=memory,
                format="markdown",
                index=None,
                max_content_length=max_content_length,
                show_artifacts=False,
                formatted_time=time_str,
            )

    def _render_index_only(self, memory: MemoryAtom, index: int, fmt: RenderFormat) -> str:
        """
        仅渲染 Index 层 (摘要视图)

        Args:
            memory: 记忆原子
            index: 索引编号
            fmt: 渲染格式

        Returns:
            渲染后的文本
        """
        summary = memory.index.summary
        if len(summary) > self.config.index_max_summary_length:
            summary = summary[:self.config.index_max_summary_length] + "..."

        hint = self._render_lazy_load_hint(memory) if self.config.enable_lazy_loading else "如需详情请询问"

        if fmt == RenderFormat.XML:
            tags = ", ".join(f"#{tag}" for tag in memory.index.tags) or "(无标签)"
            return INDEX_XML_TEMPLATE.format(
                id=index,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )
        else:
            tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
            return INDEX_MD_TEMPLATE.format(
                title=memory.index.title,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )

    def _render_lazy_load_hint(self, memory: MemoryAtom) -> str:
        """渲染懒加载工具提示"""
        if self.config.enable_lazy_loading:
            tool_name = self.config.lazy_load_tool_name
            return f'使用 {tool_name}("{memory.id}") 获取完整内容'
        return self.config.lazy_load_hint


class CompactContextRenderer(BaseContextRenderer):
    """
    紧凑上下文渲染器

    仅渲染 Index 层信息 (摘要+标签)，不渲染完整 Payload。
    适用于 Token 预算极为有限的场景，配合懒加载工具使用。
    """

    def __init__(self, config: CompactRendererConfig):
        """
        初始化紧凑渲染器

        Args:
            config: 紧凑渲染器配置
        """

        self.config = config
        self._time_formatter = TimeFormatter(language=Language.CHINESE, stale_days=90)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        """
        渲染记忆列表 (仅 Index 层)

        Args:
            results: SearchResult 列表或 MemoryAtom 列表
            render_format: 输出格式（可选，覆盖默认）

        Returns:
            渲染后的上下文字符串
        """
        if not results:
            return ""

        fmt = render_format
        if fmt is None:
            fmt = RenderFormat.XML if self.config.render_format.lower() == "xml" else RenderFormat.MARKDOWN

        memories = _extract_memories(results)
        if not memories:
            return ""

        if fmt == RenderFormat.XML:
            header, footer = XML_HEADER, XML_FOOTER_SIMPLE
        else:
            header, footer = MD_HEADER, MD_FOOTER_SIMPLE

        header_footer_tokens = estimate_tokens(header) + estimate_tokens(footer)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        rendered_blocks = []
        remaining_budget = available_budget

        for i, memory in enumerate(memories):
            index = i + 1
            index_block = self._render_index_only(memory, index, fmt)
            index_tokens = estimate_tokens(index_block)

            if index_tokens <= remaining_budget:
                rendered_blocks.append(index_block)
                remaining_budget -= index_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(rendered_blocks)} 条)")
                break

        if not rendered_blocks:
            return ""

        return header + "".join(rendered_blocks) + footer

    def _render_index_only(self, memory: MemoryAtom, index: int, fmt: RenderFormat) -> str:
        """
        仅渲染 Index 层 (摘要视图)

        Args:
            memory: 记忆原子
            index: 索引编号
            fmt: 渲染格式

        Returns:
            渲染后的文本
        """
        summary = memory.index.summary
        if len(summary) > self.config.index_max_summary_length:
            summary = summary[:self.config.index_max_summary_length] + "..."

        hint = self._render_lazy_load_hint(memory) if self.config.enable_lazy_loading else "如需详情请询问"

        if fmt == RenderFormat.XML:
            tags = ", ".join(f"#{tag}" for tag in memory.index.tags) or "(无标签)"
            return INDEX_XML_TEMPLATE.format(
                id=index,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )
        else:
            tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
            return INDEX_MD_TEMPLATE.format(
                title=memory.index.title,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )

    def _render_lazy_load_hint(self, memory: MemoryAtom) -> str:
        """渲染懒加载工具提示"""
        if self.config.enable_lazy_loading:
            tool_name = self.config.lazy_load_tool_name
            return f'使用 {tool_name}("{memory.id}") 获取完整内容'
        return self.config.lazy_load_hint


# ========== 工厂函数 ==========

def create_renderer(
    config: Union[FullRendererConfig, CascadeRendererConfig, CompactRendererConfig]
) -> BaseContextRenderer:
    """
    创建渲染器工厂

    支持多态配置:
    - FullRendererConfig -> FullContextRenderer
    - CascadeRendererConfig -> CascadeContextRenderer
    - CompactRendererConfig -> CompactContextRenderer

    Args:
        config: 渲染器配置

    Returns:
        BaseContextRenderer 实例
    """
    if isinstance(config, FullRendererConfig):
        return FullContextRenderer(config)

    if isinstance(config, CascadeRendererConfig):
        return CascadeContextRenderer(config)

    if isinstance(config, CompactRendererConfig):
        return CompactContextRenderer(config)

    raise ValueError(f"未知的渲染器配置类型: {type(config)}")


__all__ = [
    "FullContextRenderer",
    "CascadeContextRenderer",
    "CompactContextRenderer",
    "create_renderer",
]
