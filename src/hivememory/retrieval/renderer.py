"""
上下文渲染模块

职责:
    将检索到的记忆原子渲染为适合注入 LLM Context 的格式

输出格式:
    - XML 标签格式（Claude/GPT-4 推荐）
    - Markdown 格式（通用）

对应设计文档: PROJECT.md 5.2 节
"""

from typing import List, Optional
from datetime import datetime
import logging

from hivememory.core.models import MemoryAtom, VerificationStatus
from hivememory.retrieval.models import RenderFormat
from hivememory.retrieval.interfaces import ContextRenderer as ContextRendererInterface
from hivememory.utils import TimeFormatter, Language

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

    XML_MEMORY_TEMPLATE = """
<memory_block id="{id}" type="{type}">
    [标签]: {tags}
    (时间): {time}
    [置信度]: {confidence}
    [内容]:
    {content}
</memory_block>"""
    
    # Markdown 模板
    MD_HEADER = """## 相关记忆上下文

以下是与当前对话相关的历史记忆，可用于保持一致性和复用知识：

---
"""
    
    MD_FOOTER = """
---

> 如果某条记忆标记为 (Warning: Old) 或 [Unverified]，请在使用前验证。
"""
    
    MD_MEMORY_TEMPLATE = """
### 📌 {title}

- **类型**: `{type}`
- **标签**: {tags}
- **时间**: {time}
- **置信度**: {confidence}

{content}

---"""
    
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
        confidence_str = self._format_confidence(memory)
        content = self._truncate_content(memory.payload.content)

        # 根据格式确定模板和参数
        if index is not None:  # XML 格式
            tags = ", ".join(f"#{tag}" for tag in memory.index.tags)
            tags_empty = "(无标签)"

            return self.XML_MEMORY_TEMPLATE.format(
                id=index,
                type=memory.index.memory_type.value,
                tags=tags if tags else tags_empty,
                time=time_str,
                confidence=confidence_str,
                content=content
            )
        else:  # Markdown 格式
            tags = ", ".join(f"`{tag}`" for tag in memory.index.tags)
            tags_empty = "(无标签)"

            return self.MD_MEMORY_TEMPLATE.format(
                title=memory.index.title,
                type=memory.index.memory_type.value,
                tags=tags if tags else tags_empty,
                time=time_str,
                confidence=confidence_str,
                content=content
            )

    def _format_confidence(self, memory: MemoryAtom) -> str:
        """格式化置信度"""
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
    
    def _truncate_content(self, content: str) -> str:
        """截断过长的内容"""
        if len(content) <= self.max_content_length:
            return content

        # 智能截断：尝试在句子边界截断
        truncated = content[:self.max_content_length]

        # 尝试找到最后一个完整句子
        for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
            last_sep = truncated.rfind(sep)
            if last_sep > self.max_content_length // 2:
                truncated = truncated[:last_sep + len(sep)]
                break

        return truncated + "\n\n[内容已截断。如需完整内容请询问。]"


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
        from hivememory.core.config import ContextRendererConfig
        config = ContextRendererConfig()

    fmt = RenderFormat.XML if config.render_format.lower() == "xml" else RenderFormat.MARKDOWN

    return ContextRenderer(
        render_format=fmt,
        max_tokens=config.max_tokens,
        max_content_length=config.max_content_length,
        show_artifacts=config.include_artifact
    )


__all__ = [
    "ContextRenderer",
    "MinimalRenderer",
    "create_default_renderer",
]
