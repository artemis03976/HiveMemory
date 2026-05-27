"""MemoryAtom compilation handler."""

from __future__ import annotations

from hivememory.core.models import MemoryAtom, VerificationStatus
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.utils.time_formatter import Language, TimeFormatter


FULL_ITEM_TEMPLATE = """
<memory alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}

[完整内容]:
{content}
{history}
</memory>"""

INDEX_ITEM_TEMPLATE = """
<memory_index alias="{alias}">
### {title}
- **类型**: `{type}` | **存档于**: {time} | **置信度**: {confidence}
- **标签**:  {tags}
- **内容摘要**: {summary}
</memory_index>"""

AGENT_PROFILE_ITEM_TEMPLATE = """
<agent_profile alias="{alias}">
- **角色**: {title}
- **能力特长**: {summary}
</agent_profile>"""


def compile_memory_atom(
    atom: MemoryAtom,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a single MemoryAtom into the requested target."""
    alias = atom.get_alias()
    effective_alias = options.requested_alias or alias

    if target == MemoryCompileTarget.PROMPT_FULL:
        text = _render_full_context(
            atom,
            max_content_length=options.max_content_length,
            stale_days=options.stale_days,
        )
    elif target == MemoryCompileTarget.PROMPT_INDEX:
        text = _render_index_context(
            atom,
            max_summary_length=options.max_summary_length,
            stale_days=options.stale_days,
        )
    elif target == MemoryCompileTarget.DENSE_EMBEDDING:
        text = _render_dense_embedding(atom)
    elif target == MemoryCompileTarget.SPARSE_EMBEDDING:
        text = _render_sparse_embedding(atom)
    elif target == MemoryCompileTarget.AGENT_PROFILE_MENU:
        text = _render_agent_profile(atom)
    elif target == MemoryCompileTarget.MTP_READ:
        text = f"[{effective_alias}]:\n{atom.payload.content}"
    elif target == MemoryCompileTarget.SHARED_CONTEXT:
        text = f"[{effective_alias}]:\n{atom.payload.content}"
    elif target == MemoryCompileTarget.RUNNABLE_TOOL:
        raise ValueError("RUNNABLE_TOOL target is reserved for Phase 3.")
    else:
        raise ValueError(f"Unsupported target '{target}' for MemoryAtom source.")

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind="atom",
        alias=alias,
        memory_id=str(atom.id),
    )


def _render_dense_embedding(memory: MemoryAtom) -> str:
    return (
        f"Title: {memory.index.title}\n"
        f"Type: {memory.index.memory_type.value}\n"
        f"Tags: {', '.join(memory.index.tags)}\n"
        f"Summary: {memory.index.summary}"
    )


def _render_sparse_embedding(memory: MemoryAtom) -> str:
    tags_string = " ".join(memory.index.tags)
    return (
        f"{memory.index.title} {memory.index.title} "
        f"{tags_string} {tags_string} "
        f"{memory.index.summary}"
    )


def _render_full_context(
    memory: MemoryAtom,
    max_content_length: int = 500,
    stale_days: int = 90,
) -> str:
    content = _truncate_content(memory.payload.content, max_content_length)
    confidence_str = _format_confidence(memory)
    alias = memory.get_alias()
    tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
    time_str = TimeFormatter(
        language=Language.CHINESE,
        stale_days=stale_days,
    ).format(memory.meta.updated_at)

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


def _render_index_context(
    memory: MemoryAtom,
    max_summary_length: int = 100,
    stale_days: int = 90,
) -> str:
    alias = memory.get_alias()
    confidence_str = _format_confidence(memory)
    tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or "(无标签)"
    time_str = TimeFormatter(
        language=Language.CHINESE,
        stale_days=stale_days,
    ).format(memory.meta.updated_at)

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


def _format_confidence(memory: MemoryAtom) -> str:
    score = memory.meta.confidence_score
    status = memory.meta.verification_status

    status_str = ""
    if status == VerificationStatus.VERIFIED:
        status_str = " [已验证]"
    elif status == VerificationStatus.DEPRECATED:
        status_str = " [已废弃]"
    elif status == VerificationStatus.HALLUCINATION:
        status_str = " [警告：幻觉]"
    elif score < 0.7:
        status_str = " [未验证]"

    if score >= 0.9:
        return f"{score:.0%} (高){status_str}"
    if score >= 0.7:
        return f"{score:.0%} (中){status_str}"
    return f"{score:.0%} (低){status_str}"


def _truncate_content(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]

    for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_length // 2:
            truncated = truncated[:last_sep + len(sep)]
            break

    return truncated + "\n\n[...部分内容已截断，如需阅读完整内容请使用 READ 指令读取...]"


def _render_agent_profile(memory: MemoryAtom) -> str:
    alias = memory.get_alias()
    title = memory.index.title if memory.index.title else "(未命名子代理)"
    summary = memory.index.summary if memory.index.summary else ""

    return AGENT_PROFILE_ITEM_TEMPLATE.format(
        alias=alias,
        title=title,
        summary=summary,
    )
