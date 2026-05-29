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
from hivememory.i18n import (
    get_memory_envelope_text,
)

logger = logging.getLogger(__name__)

_compiler = MemoryCompiler()

_EMPTY_CONTEXT_NOTICE = get_memory_envelope_text("retrieval_empty_context_notice", "zh")
_MEMORY_EMPTY_HINT = get_memory_envelope_text("retrieval_memory_empty_hint", "zh")
_AGENT_EMPTY_HINT = get_memory_envelope_text("retrieval_agent_empty_hint", "zh")


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


class _RendererI18nMixin:
    def _init_i18n(self, default_language: str = "zh") -> None:
        self.default_language = default_language
        self._compiler = MemoryCompiler(default_language=default_language)

    def _text(self, key: str) -> str:
        return get_memory_envelope_text(key, self.default_language)

    @property
    def _retrieval_header(self) -> str:
        return get_memory_envelope_text("retrieval_header", self.default_language)

    @property
    def _retrieval_footer(self) -> str:
        return get_memory_envelope_text("retrieval_footer", self.default_language)

    def _retrieval_envelope_length(self) -> int:
        return len(self._retrieval_header) + len(self._retrieval_footer)

    def _retrieval_envelope_tokens(self) -> int:
        return estimate_tokens(self._retrieval_header) + estimate_tokens(self._retrieval_footer)

    @property
    def _empty_context_notice(self) -> str:
        return self._text("retrieval_empty_context_notice")

    @property
    def _memory_empty_hint(self) -> str:
        return self._text("retrieval_memory_empty_hint")

    @property
    def _agent_empty_hint(self) -> str:
        return self._text("retrieval_agent_empty_hint")

    def _compile_agent_profile_artifacts(
        self,
        agents: List[MemoryAtom],
    ) -> List[CompiledMemoryArtifact]:
        return [
            self._compiler.compile(agent, MemoryCompileTarget.AGENT_PROFILE_MENU)
            for agent in agents
        ]

    def _wrap_retrieval_context(
        self,
        memory_artifacts: List[CompiledMemoryArtifact],
        agent_artifacts: List[CompiledMemoryArtifact],
    ) -> str:
        sections = [
            MemoryEnvelopeSection(
                kind="memories",
                artifacts=memory_artifacts,
                empty_text=self._memory_empty_hint if agent_artifacts else None,
            ),
            MemoryEnvelopeSection(
                kind="agent_profiles",
                artifacts=agent_artifacts,
                empty_text=self._agent_empty_hint if memory_artifacts else None,
            ),
        ]
        return self._compiler.wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=sections,
        ).text


class FullContextRenderer(_RendererI18nMixin, BaseContextRenderer):
    """Render all memories as full context, truncating content by character limit."""

    def __init__(self, config: FullRendererConfig, default_language: str = "zh"):
        self.config = config
        self.max_tokens = config.max_tokens
        self.max_content_length = config.max_content_length
        self.show_artifacts = config.show_artifacts
        self.stale_days = config.stale_days
        self._init_i18n(default_language)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        all_memories = _extract_memories(results) if results else []
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        memory_artifacts: list[CompiledMemoryArtifact] = []
        total_length = self._retrieval_envelope_length()

        for memory in memories:
            artifact = self._render_memory(memory)
            block = artifact.text

            if total_length + len(block) > self.max_tokens:
                logger.debug(f"达到长度限制，截断至 {len(memory_artifacts)} 条记忆")
                break

            memory_artifacts.append(artifact)
            total_length += len(block)

        agent_artifacts = self._compile_agent_profile_artifacts(agent_profiles)

        # 场景 1: 两者均空，返回精简闭环提示
        if not memory_artifacts and not agent_artifacts:
            return self._empty_context_notice

        return self._wrap_retrieval_context(memory_artifacts, agent_artifacts)

    def _render_memory(self, memory: MemoryAtom) -> CompiledMemoryArtifact:
        return self._compiler.compile(
            memory,
            MemoryCompileTarget.PROMPT_FULL,
            MemoryCompileOptions(
                max_content_length=self.max_content_length,
                stale_days=self.stale_days,
            ),
        )


class CascadeContextRenderer(_RendererI18nMixin, BaseContextRenderer):
    """Render top memories fully, then degrade the rest to index context."""

    def __init__(self, config: CascadeRendererConfig, default_language: str = "zh"):
        self.config = config
        self._init_i18n(default_language)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None,
    ) -> str:
        all_memories = _extract_memories(results) if results else []
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        header_footer_tokens = self._retrieval_envelope_tokens()
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token budget is too small for memory header/footer templates")
            return ""

        rendered_artifacts, _ = self._render_with_budget(memories, available_budget)
        agent_artifacts = self._compile_agent_profile_artifacts(agent_profiles)

        if not rendered_artifacts and not agent_artifacts:
            return self._empty_context_notice

        return self._wrap_retrieval_context(rendered_artifacts, agent_artifacts)

    def _render_with_budget(
        self,
        memories: List[MemoryAtom],
        budget: int,
    ) -> Tuple[List[CompiledMemoryArtifact], int]:
        rendered_artifacts: list[CompiledMemoryArtifact] = []
        remaining_budget = budget

        for index, memory in enumerate(memories):
            if index < self.config.full_payload_count:
                full_artifact = self._compiler.compile(
                    memory,
                    MemoryCompileTarget.PROMPT_FULL,
                    MemoryCompileOptions(max_content_length=self.config.max_content_length),
                )
                full_tokens = estimate_tokens(full_artifact.text)

                if full_tokens <= remaining_budget:
                    rendered_artifacts.append(full_artifact)
                    remaining_budget -= full_tokens
                    continue

            index_artifact = self._compiler.compile(
                memory,
                MemoryCompileTarget.PROMPT_INDEX,
                MemoryCompileOptions(max_summary_length=self.config.index_max_summary_length),
            )
            index_tokens = estimate_tokens(index_artifact.text)

            if index_tokens <= remaining_budget:
                rendered_artifacts.append(index_artifact)
                remaining_budget -= index_tokens
            else:
                logger.debug("Token budget exhausted after %s memories", len(rendered_artifacts))
                break

        return rendered_artifacts, remaining_budget


class CompactContextRenderer(_RendererI18nMixin, BaseContextRenderer):
    """
    紧凑上下文渲染器

    仅渲染 Index 层信息 (摘要+标签)，不渲染完整 Payload。
    适用于 Token 预算极其有限的场景，配合 retrieval envelope 中的 READ 指令实现懒加载。
    """

    def __init__(self, config: CompactRendererConfig, default_language: str = "zh"):
        self.config = config
        self._init_i18n(default_language)

    def render(
        self,
        results: List,
        render_format: Optional[RenderFormat] = None
    ) -> str:
        all_memories = _extract_memories(results) if results else []
        memories, agent_profiles = _separate_agent_profiles(all_memories)

        header_footer_tokens = self._retrieval_envelope_tokens()
        available_budget = self.config.max_memory_tokens - header_footer_tokens

        if available_budget <= 0:
            logger.warning("Token 预算不足以容纳头尾模板")
            return ""

        artifacts: list[CompiledMemoryArtifact] = []
        remaining_budget = available_budget

        for memory in memories:
            artifact = self._compiler.compile(
                memory,
                MemoryCompileTarget.PROMPT_INDEX,
                MemoryCompileOptions(max_summary_length=self.config.index_max_summary_length),
            )
            block_tokens = estimate_tokens(artifact.text)

            if block_tokens <= remaining_budget:
                artifacts.append(artifact)
                remaining_budget -= block_tokens
            else:
                logger.debug(f"预算耗尽，停止渲染 (已渲染 {len(artifacts)} 条)")
                break

        agent_artifacts = self._compile_agent_profile_artifacts(agent_profiles)

        # 场景 1: 两者均空，返回精简闭环提示
        if not artifacts and not agent_artifacts:
            return self._empty_context_notice

        return self._wrap_retrieval_context(artifacts, agent_artifacts)


# ========== 工厂函数 ==========

def create_renderer(
    config: Union[FullRendererConfig, CascadeRendererConfig, CompactRendererConfig],
    default_language: str = "zh",
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
        return FullContextRenderer(config, default_language=default_language)

    if isinstance(config, CascadeRendererConfig):
        return CascadeContextRenderer(config, default_language=default_language)

    if isinstance(config, CompactRendererConfig):
        return CompactContextRenderer(config, default_language=default_language)

    raise ValueError(f"未知的渲染器配置类型: {type(config)}")


__all__ = [
    "_EMPTY_CONTEXT_NOTICE",
    "_MEMORY_EMPTY_HINT",
    "_AGENT_EMPTY_HINT",
    "FullContextRenderer",
    "CascadeContextRenderer",
    "CompactContextRenderer",
    "create_renderer",
]
