"""
上下文渲染模块

职责:
    将检索到的记忆原子渲染为适合注入 LLM Context 的格式

渲染器类型:
    - FullContextRenderer: 完整渲染，超过字符上限则截断
    - CascadeContextRenderer: 瀑布式渲染，Top-N 完整 + 其余 Index
    - CompactContextRenderer: 仅渲染 Index 层信息

模板统一由 memory_atom_renderer 管理。

对应设计文档: PROJECT.md 5.2 节
"""

from typing import List, Optional, Tuple, Union
import logging

from hivememory.patchouli.config import FullRendererConfig, CascadeRendererConfig, CompactRendererConfig
from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.engines.retrieval.models import RenderFormat
from hivememory.engines.retrieval.interfaces import BaseContextRenderer
from hivememory.utils import estimate_tokens
from hivememory.utils.memory_atom_renderer import (
    MemoryAtomRenderer,
    MEMORY_HEADER,
    MEMORY_FOOTER,
)

logger = logging.getLogger(__name__)


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


def _separate_agent_profiles(
    memories: List[MemoryAtom],
) -> tuple[List[MemoryAtom], List[MemoryAtom]]:
    """
    分离 AGENT_PROFILE 和普通记忆 (Phase 2: 子代理服务发现)

    Returns:
        (regular_memories, agent_profiles)
    """
    regular = []
    agents = []
    for m in memories:
        if hasattr(m, 'index') and hasattr(m.index, 'memory_type') and m.index.memory_type == MemoryType.AGENT_PROFILE:
            agents.append(m)
        else:
            regular.append(m)
    return regular, agents


def _render_agent_menu(agents: List[MemoryAtom]) -> str:
    """
    渲染子代理服务菜单 (Phase 2)

    格式:
    [Available Sub-Agents (Ready to CALL)]
    1. [ID: coder_doll] "Backend Developer" - 擅长 Python/FastAPI 后端开发
    2. [ID: translator_doll] "EN Translator" - 中英互译专家

    Args:
        agents: AGENT_PROFILE 类型的记忆原子列表

    Returns:
        str: 渲染后的菜单文本
    """
    if not agents:
        return ""

    lines = ["\n[Available Sub-Agents (Ready to CALL)]"]
    for i, agent in enumerate(agents, 1):
        alias = agent.index.alias if hasattr(agent.index, 'alias') and agent.index.alias else f"agent_{i}"
        title = agent.index.title if hasattr(agent.index, 'title') and agent.index.title else "(untitled)"
        summary = ""
        if hasattr(agent.index, 'summary') and agent.index.summary:
            summary = agent.index.summary[:80]
        lines.append(f'{i}. [ID: {alias}] "{title}" - {summary}')
    return "\n".join(lines)


class FullContextRenderer(BaseContextRenderer):
    """
    完整上下文渲染器

    渲染所有 MemoryAtom 的完整内容，超过字符上限则直接截断。
    """
    def __init__(self, config: FullRendererConfig):
        self.config = config
        self.max_tokens = config.max_tokens
        self.max_content_length = config.max_content_length
        self.show_artifacts = config.show_artifacts
        self.stale_days = config.stale_days

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        if not results:
            return ""

        memories = _extract_memories(results)
        if not memories:
            return ""

        blocks = [MEMORY_HEADER]
        total_length = len(MEMORY_HEADER) + len(MEMORY_FOOTER)

        for memory in memories:
            block = self._render_memory(memory)

            if total_length + len(block) > self.max_tokens:
                logger.debug(f"达到长度限制，截断至 {len(blocks) - 1} 条记忆")
                break

            blocks.append(block)
            total_length += len(block)

        blocks.append(MEMORY_FOOTER)
        return "".join(blocks)

    def _render_memory(self, memory: MemoryAtom) -> str:
        return MemoryAtomRenderer.for_full_context(
            memory=memory,
            max_content_length=self.max_content_length,
            stale_days=self.stale_days,
        )


class CascadeContextRenderer(BaseContextRenderer):
    """
    瀑布式上下文渲染器

    依次完整渲染 MemoryAtom，直到 Token 预算紧张时降级为 Index 层信息:
    1. Top-N 记忆强制完整渲染 (Payload)
    2. 其余按预算瀑布式降级为 Index 视图 (摘要+标签)
    3. 预算耗尽时停止渲染
    """

    def __init__(self, config: CascadeRendererConfig):
        self.config = config

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        if not results:
            return ""

        all_memories = _extract_memories(results)
        if not all_memories:
            return ""

        # Phase 2: 分离 AGENT_PROFILE 和普通记忆
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        header_footer_tokens = estimate_tokens(MEMORY_HEADER) + estimate_tokens(MEMORY_FOOTER)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        rendered_blocks, _ = self._render_with_budget(memories, available_budget)

        if not rendered_blocks and not agent_profiles:
            return ""

        result = MEMORY_HEADER + "".join(rendered_blocks)

        # Phase 2: 追加子代理服务菜单
        if agent_profiles:
            result += _render_agent_menu(agent_profiles)

        result += MEMORY_FOOTER
        return result

    def _render_with_budget(
        self,
        memories: List[MemoryAtom],
        budget: int,
    ) -> Tuple[List[str], int]:
        rendered_blocks = []
        remaining_budget = budget

        for i, memory in enumerate(memories):
            # Top-N 强制完整渲染
            if i < self.config.full_payload_count:
                full_block = MemoryAtomRenderer.for_full_context(
                    memory=memory,
                    max_content_length=self.config.max_content_length,
                )
                full_tokens = estimate_tokens(full_block)

                if full_tokens <= remaining_budget:
                    rendered_blocks.append(full_block)
                    remaining_budget -= full_tokens
                    continue
                # 预算不足，降级为 Index

            # 尝试 Index 视图渲染
            index_block = MemoryAtomRenderer.for_index_context(
                memory=memory,
                max_summary_length=self.config.index_max_summary_length,
            )
            index_tokens = estimate_tokens(index_block)

            if index_tokens <= remaining_budget:
                rendered_blocks.append(index_block)
                remaining_budget -= index_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(rendered_blocks)} 条)")
                break

        return rendered_blocks, remaining_budget


class CompactContextRenderer(BaseContextRenderer):
    """
    紧凑上下文渲染器

    仅渲染 Index 层信息 (摘要+标签)，不渲染完整 Payload。
    适用于 Token 预算极其有限的场景，配合 MEMORY_FOOTER 中的 READ 指令实现懒加载。
    """

    def __init__(self, config: CompactRendererConfig):
        self.config = config

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        if not results:
            return ""

        memories = _extract_memories(results)
        if not memories:
            return ""

        header_footer_tokens = estimate_tokens(MEMORY_HEADER) + estimate_tokens(MEMORY_FOOTER)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        blocks = []
        remaining_budget = available_budget

        for memory in memories:
            block = MemoryAtomRenderer.for_index_context(
                memory=memory,
                max_summary_length=self.config.index_max_summary_length,
            )
            block_tokens = estimate_tokens(block)

            if block_tokens <= remaining_budget:
                blocks.append(block)
                remaining_budget -= block_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(blocks)} 条)")
                break

        if not blocks:
            return ""

        return MEMORY_HEADER + "".join(blocks) + MEMORY_FOOTER


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
