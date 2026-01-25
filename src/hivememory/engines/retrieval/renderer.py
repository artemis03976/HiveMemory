"""
上下文渲染模块

职责:
    将检索到的记忆原子渲染为适合注入 LLM Context 的格式

输出格式:
    - XML 标签格式（Claude/GPT-4 推荐）
    - Markdown 格式（通用）

对应设计文档: PROJECT.md 5.2 节
"""

from typing import List, Optional, Tuple
import logging

from hivememory.core.models import MemoryAtom, estimate_tokens
from hivememory.engines.retrieval.models import RenderFormat
from hivememory.engines.retrieval.interfaces import BaseContextRenderer as ContextRendererInterface
from hivememory.utils import TimeFormatter, Language, MemoryAtomRenderer

logger = logging.getLogger(__name__)


class ContextRenderer(ContextRendererInterface):
    """
    上下文渲染器
    
    将记忆原子列表渲染为 LLM 可读的格式
    """
    
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

    # Markdown 模板
    MD_HEADER = """## 相关记忆上下文

以下是与当前对话相关的历史记忆，可用于保持一致性和复用知识：

---
"""
    
    MD_FOOTER = """
---

> 如果某条记忆标记为 (Warning: Old) 或 [Unverified]，请在使用前验证。
"""

    def __init__(
        self,
        render_format: RenderFormat = RenderFormat.XML,
        max_tokens: int = 2000,
        max_content_length: int = 500,
        show_artifacts: bool = False,
        language: Language = Language.CHINESE,
        stale_days: int = 90,
    ):
        """
        初始化渲染器

        Args:
            render_format: 输出格式（XML 或 Markdown）
            max_tokens: 最大输出长度（字符数估算）
            max_content_length: 单条记忆的最大内容长度
            show_artifacts: 是否显示原始数据链接
            language: 时间格式化语言（默认中文）
            stale_days: 超过此天数显示陈旧警告（默认90天）
        """
        self.render_format = render_format
        self.max_tokens = max_tokens
        self.max_content_length = max_content_length
        self.show_artifacts = show_artifacts
        self._time_formatter = TimeFormatter(language=language, stale_days=stale_days)
    
    def render(
        self,
        results: List,  # SearchResult or MemoryAtom list
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
        render_format = render_format or self.render_format
        
        if not results:
            return ""
        
        # 统一转换为 MemoryAtom 列表
        memories = []
        for item in results:
            if hasattr(item, 'memory'):
                memories.append(item.memory)
            elif isinstance(item, MemoryAtom):
                memories.append(item)
            else:
                logger.warning(f"未知的结果类型: {type(item)}")
        
        if not memories:
            return ""
        
        # 根据格式选择渲染方法
        if render_format == RenderFormat.XML:
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

            # 检查长度限制
            if total_length + len(block) > self.max_tokens:
                logger.debug(f"达到长度限制，截断至 {i-1} 条记忆")
                break

            blocks.append(block)
            total_length += len(block)

        blocks.append(footer)
        return "".join(blocks)

    def _render_xml(self, memories: List[MemoryAtom]) -> str:
        """渲染为 XML 格式"""
        return self._render_format(memories, self.XML_HEADER, self.XML_FOOTER, use_index=True)

    def _render_markdown(self, memories: List[MemoryAtom]) -> str:
        """渲染为 Markdown 格式"""
        return self._render_format(memories, self.MD_HEADER, self.MD_FOOTER, use_index=False)
    
    def _render_memory(self, memory: MemoryAtom, index: Optional[int] = None) -> str:
        """
        通用单条记忆渲染函数

        Args:
            memory: 记忆原子
            index: 索引编号（XML格式需要，Markdown为None）
        """
        # 使用 TimeFormatter 格式化时间
        time_str = self._time_formatter.format(memory.meta.updated_at)

        # 使用 MemoryAtomRenderer 进行渲染
        format_type = "xml" if index is not None else "markdown"
        return MemoryAtomRenderer.for_llm_context(
            memory=memory,
            format=format_type,
            index=index,
            max_content_length=self.max_content_length,
            show_artifacts=self.show_artifacts,
            formatted_time=time_str,
        )


class MinimalRenderer(ContextRendererInterface):
    """
    极简渲染器

    仅输出核心信息，最小化 Token 消耗
    """

    def render(self, results: List, render_format: Optional[RenderFormat] = None) -> str:
        """渲染为紧凑格式"""
        if not results:
            return ""

        lines = ["[相关记忆]"]

        for i, item in enumerate(results[:5], 1):
            memory = item.memory if hasattr(item, 'memory') else item
            tags = ",".join(memory.index.tags[:3])
            preview = memory.payload.content[:100].replace("\n", " ")
            lines.append(f"{i}. [{tags}] {memory.index.title}: {preview}...")

        return "\n".join(lines)


def create_default_renderer(config: Optional["ContextRendererConfig"] = None) -> ContextRenderer:
    """
    创建默认渲染器

    Args:
        config: 上下文渲染配置

    Returns:
        ContextRenderer 实例
    """
    if config is None:
        from hivememory.patchouli.config import ContextRendererConfig
        config = ContextRendererConfig()

    fmt = RenderFormat.XML if config.render_format.lower() == "xml" else RenderFormat.MARKDOWN

    return ContextRenderer(
        render_format=fmt,
        max_tokens=config.max_tokens,
        max_content_length=config.max_content_length,
        show_artifacts=config.include_artifact
    )


class CompactContextRenderer(ContextRendererInterface):
    """
    紧凑上下文渲染器

    实现 Token 预算管理和分级渲染:
    1. Top-N 记忆强制完整渲染 (Payload)
    2. 其余按预算瀑布式降级为 Index 视图 (摘要+标签)
    3. 预算耗尽时停止渲染

    与 ContextRenderer 的区别:
    - ContextRenderer: 简单的字符数截断
    - CompactContextRenderer: 智能的分级渲染，优先保证重要记忆的完整性
    """

    # Index 视图 XML 模板
    INDEX_XML_TEMPLATE = """
<memory_ref id="{id}" type="{type}">
    [标签]: {tags}
    [摘要]: {summary}
    [提示]: {hint}
</memory_ref>"""

    # Index 视图 Markdown 模板
    INDEX_MD_TEMPLATE = """
### 📎 {title} (摘要)

- **类型**: `{type}`
- **标签**: {tags}
- **摘要**: {summary}

> {hint}

---"""

    # 头部模板
    XML_HEADER = """<system_memory_context>
以下是从历史交互中检索到的相关记忆。
使用这些记忆来保持一致性并复用已有知识。
"""

    XML_FOOTER = """
</system_memory_context>
"""

    MD_HEADER = """## 相关记忆上下文

以下是与当前对话相关的历史记忆：

---
"""

    MD_FOOTER = """
---
"""

    def __init__(self, config: Optional["CompactRendererConfig"] = None):
        """
        初始化紧凑渲染器

        Args:
            config: 紧凑渲染器配置
        """
        if config is None:
            from hivememory.patchouli.config import CompactRendererConfig
            config = CompactRendererConfig()

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

        # 确定渲染格式
        fmt = render_format
        if fmt is None:
            fmt = RenderFormat.XML if self.config.render_format.lower() == "xml" else RenderFormat.MARKDOWN

        # 统一转换为 MemoryAtom 列表
        memories = self._extract_memories(results)
        if not memories:
            return ""

        # 选择头尾模板
        if fmt == RenderFormat.XML:
            header, footer = self.XML_HEADER, self.XML_FOOTER
        else:
            header, footer = self.MD_HEADER, self.MD_FOOTER

        # 计算可用预算
        header_footer_tokens = self._estimate_tokens(header) + self._estimate_tokens(footer)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        # 执行分级渲染
        rendered_blocks, _ = self._render_with_budget(memories, available_budget, fmt)

        if not rendered_blocks:
            return ""

        # 组装最终输出
        return header + "".join(rendered_blocks) + footer

    def _extract_memories(self, results: List) -> List[MemoryAtom]:
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

            # 判断是否强制完整渲染
            force_full = (i < self.config.full_payload_count) if self.config.enable_tiered_rendering else True

            if force_full:
                # 尝试完整渲染
                full_block = self._render_full_payload(memory, index, fmt)
                full_tokens = self._estimate_tokens(full_block)

                if full_tokens <= remaining_budget:
                    rendered_blocks.append(full_block)
                    remaining_budget -= full_tokens
                    continue
                else:
                    # 预算不足，尝试降级为 Index
                    if not self.config.enable_tiered_rendering:
                        # 不启用分级渲染，直接停止
                        logger.debug(f"预算不足，停止渲染 (已渲染 {len(rendered_blocks)} 条)")
                        break

            # 尝试 Index 视图渲染
            index_block = self._render_index_only(memory, index, fmt)
            index_tokens = self._estimate_tokens(index_block)

            if index_tokens <= remaining_budget:
                rendered_blocks.append(index_block)
                remaining_budget -= index_tokens
            else:
                # 预算耗尽，停止渲染
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(rendered_blocks)} 条)")
                break

        return rendered_blocks, remaining_budget

    def _render_full_payload(self, memory: MemoryAtom, index: int, fmt: RenderFormat) -> str:
        """
        渲染完整 Payload

        复用 MemoryAtomRenderer 的渲染逻辑

        Args:
            memory: 记忆原子
            index: 索引编号
            fmt: 渲染格式

        Returns:
            渲染后的文本
        """
        time_str = self._time_formatter.format(memory.meta.updated_at)

        if fmt == RenderFormat.XML:
            # XML 格式使用 index
            return MemoryAtomRenderer.for_llm_context(
                memory=memory,
                format="xml",
                index=index,
                max_content_length=500,
                show_artifacts=False,
                formatted_time=time_str,
            )
        else:
            # Markdown 格式不使用 index (传 None)
            return MemoryAtomRenderer.for_llm_context(
                memory=memory,
                format="markdown",
                index=None,
                max_content_length=500,
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
        # 截断摘要
        summary = memory.index.summary
        if len(summary) > self.config.index_max_summary_length:
            summary = summary[:self.config.index_max_summary_length] + "..."

        # 构建提示文本
        hint = self._render_lazy_load_hint(memory) if self.config.enable_lazy_loading else "如需详情请询问"

        if fmt == RenderFormat.XML:
            tags = ", ".join(f"#{tag}" for tag in memory.index.tags) or "(无标签)"
            return self.INDEX_XML_TEMPLATE.format(
                id=index,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )
        else:
            tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
            return self.INDEX_MD_TEMPLATE.format(
                title=memory.index.title,
                type=memory.index.memory_type.value,
                tags=tags,
                summary=summary,
                hint=hint,
            )

    def _render_lazy_load_hint(self, memory: MemoryAtom) -> str:
        """
        渲染懒加载工具提示

        Args:
            memory: 记忆原子

        Returns:
            提示文本
        """
        if self.config.enable_lazy_loading:
            tool_name = self.config.lazy_load_tool_name
            return f'使用 {tool_name}("{memory.id}") 获取完整内容'
        return self.config.lazy_load_hint

    def _estimate_tokens(self, text: str) -> int:
        """
        估算 Token 数量

        复用 core/models.py 中的 estimate_tokens 函数

        Args:
            text: 文本

        Returns:
            估算的 Token 数量
        """
        return estimate_tokens(text)


def create_compact_renderer(config: Optional["CompactRendererConfig"] = None) -> CompactContextRenderer:
    """
    创建紧凑渲染器

    Args:
        config: 紧凑渲染器配置

    Returns:
        CompactContextRenderer 实例
    """
    return CompactContextRenderer(config)


__all__ = [
    "ContextRenderer",
    "MinimalRenderer",
    "CompactContextRenderer",
    "create_default_renderer",
    "create_compact_renderer",
]
