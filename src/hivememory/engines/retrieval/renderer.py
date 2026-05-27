"""
上下文渲染模块

职责:
    将检索到的记忆原子渲染为适合注入 LLM Context 的格式

渲染器类型:
    - FullContextRenderer: 完整渲染，超过字符上限则截断
    - CascadeContextRenderer: 瀑布式渲染，Top-N 完整 + 其余 Index
    - CompactContextRenderer: 仅渲染 Index 层信息

单项记忆由 MemoryCompiler 编译，整体上下文由 retrieval envelope 包装。

对应设计文档: PROJECT.md 5.2 节
"""

from typing import List, Optional, Tuple, Union
import logging

from hivememory.system.config import FullRendererConfig, CascadeRendererConfig, CompactRendererConfig
from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.engines.retrieval.models import RenderFormat
from hivememory.engines.retrieval.interfaces import BaseContextRenderer
from hivememory.utils import estimate_tokens
from hivememory.engines.memory_compiler import (
    CompiledMemoryArtifact,
    MemoryCompiler,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.engines.memory_compiler.envelope_templates import (
    MEMORY_FOOTER,
    MEMORY_HEADER,
)

logger = logging.getLogger(__name__)

_compiler = MemoryCompiler()


# ========== 空结果提示 ==========

_EMPTY_CONTEXT_NOTICE = (
    "[System Guidance]: 帕秋莉在本次预检索中未发现强相关的历史记忆或子代理。\n"
    "(提示: 如果你需要了解历史记忆或寻找特定帮手，请随时使用 ⟪ SEARCH ⟫ 协议指令进行全局模糊搜索。)"
)

_MEMORY_EMPTY_HINT = "当前检索结果为空。若需查阅历史记忆，请使用 ⟪ SEARCH ⟫。"

_AGENT_EMPTY_HINT = '当前未发现相关的专业子代理。若需其他代理协助，请使用 ⟪ SEARCH | * | filter="type:AGENT_PROFILE" ⟫。'

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


def _compile_agent_profile_artifacts(agents: List[MemoryAtom]) -> List[CompiledMemoryArtifact]:
    artifacts: list[CompiledMemoryArtifact] = []
    for agent in agents:
        artifacts.append(_compiler.compile(agent, MemoryCompileTarget.AGENT_PROFILE_MENU))
    return artifacts


def _wrap_retrieval_context(
    memory_artifacts: List[CompiledMemoryArtifact],
    agent_artifacts: List[CompiledMemoryArtifact],
) -> str:
    sections = [
        MemoryEnvelopeSection(
            kind="memories",
            artifacts=memory_artifacts,
            empty_text=_MEMORY_EMPTY_HINT if agent_artifacts else None,
        ),
        MemoryEnvelopeSection(
            kind="agent_profiles",
            artifacts=agent_artifacts,
            empty_text=_AGENT_EMPTY_HINT if memory_artifacts else None,
        ),
    ]
    return _compiler.wrap(
        envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        sections=sections,
    ).text


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
        all_memories = _extract_memories(results) if results else []
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        memory_artifacts: list[CompiledMemoryArtifact] = []
        total_length = len(MEMORY_HEADER) + len(MEMORY_FOOTER)

        for memory in memories:
            artifact = self._render_memory(memory)
            block = artifact.text

            if total_length + len(block) > self.max_tokens:
                logger.debug(f"达到长度限制，截断至 {len(memory_artifacts)} 条记忆")
                break

            memory_artifacts.append(artifact)
            total_length += len(block)

        agent_artifacts = _compile_agent_profile_artifacts(agent_profiles)

        # 场景 1: 两者均空，返回精简闭环提示
        if not memory_artifacts and not agent_artifacts:
            return _EMPTY_CONTEXT_NOTICE

        return _wrap_retrieval_context(memory_artifacts, agent_artifacts)

    def _render_memory(self, memory: MemoryAtom) -> CompiledMemoryArtifact:
        return _compiler.compile(
            memory, MemoryCompileTarget.PROMPT_FULL,
            MemoryCompileOptions(
                max_content_length=self.max_content_length,
                stale_days=self.stale_days,
            ),
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
        all_memories = _extract_memories(results) if results else []

        # Phase 2: 分离 AGENT_PROFILE 和普通记忆
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        header_footer_tokens = estimate_tokens(MEMORY_HEADER) + estimate_tokens(MEMORY_FOOTER)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        rendered_artifacts, _ = self._render_with_budget(memories, available_budget)
        agent_artifacts = _compile_agent_profile_artifacts(agent_profiles)

        # 场景 1: 两者均空，返回精简闭环提示
        if not rendered_artifacts and not agent_artifacts:
            return _EMPTY_CONTEXT_NOTICE

        return _wrap_retrieval_context(rendered_artifacts, agent_artifacts)

    def _render_with_budget(
        self,
        memories: List[MemoryAtom],
        budget: int,
    ) -> Tuple[List[CompiledMemoryArtifact], int]:
        rendered_artifacts: list[CompiledMemoryArtifact] = []
        remaining_budget = budget

        for i, memory in enumerate(memories):
            # Top-N 强制完整渲染
            if i < self.config.full_payload_count:
                full_artifact = _compiler.compile(
                    memory, MemoryCompileTarget.PROMPT_FULL,
                    MemoryCompileOptions(max_content_length=self.config.max_content_length),
                )
                full_block = full_artifact.text
                full_tokens = estimate_tokens(full_block)

                if full_tokens <= remaining_budget:
                    rendered_artifacts.append(full_artifact)
                    remaining_budget -= full_tokens
                    continue
                # 预算不足，降级为 Index

            # 尝试 Index 视图渲染
            index_artifact = _compiler.compile(
                memory, MemoryCompileTarget.PROMPT_INDEX,
                MemoryCompileOptions(max_summary_length=self.config.index_max_summary_length),
            )
            index_block = index_artifact.text
            index_tokens = estimate_tokens(index_block)

            if index_tokens <= remaining_budget:
                rendered_artifacts.append(index_artifact)
                remaining_budget -= index_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(rendered_artifacts)} 条)")
                break

        return rendered_artifacts, remaining_budget


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
        all_memories = _extract_memories(results) if results else []
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        header_footer_tokens = estimate_tokens(MEMORY_HEADER) + estimate_tokens(MEMORY_FOOTER)
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        artifacts: list[CompiledMemoryArtifact] = []
        remaining_budget = available_budget

        for memory in memories:
            artifact = _compiler.compile(
                memory, MemoryCompileTarget.PROMPT_INDEX,
                MemoryCompileOptions(max_summary_length=self.config.index_max_summary_length),
            )
            block = artifact.text
            block_tokens = estimate_tokens(block)

            if block_tokens <= remaining_budget:
                artifacts.append(artifact)
                remaining_budget -= block_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(artifacts)} 条)")
                break

        agent_artifacts = _compile_agent_profile_artifacts(agent_profiles)

        # 场景 1: 两者均空，返回精简闭环提示
        if not artifacts and not agent_artifacts:
            return _EMPTY_CONTEXT_NOTICE

        return _wrap_retrieval_context(artifacts, agent_artifacts)


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
