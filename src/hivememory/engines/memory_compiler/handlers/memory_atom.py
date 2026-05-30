"""MemoryAtom compilation handler."""

from __future__ import annotations

from hivememory.core.models import MemoryAtom, VerificationStatus
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n import get_memory_atom_text
from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language as I18nLanguage
from hivememory.utils.time_formatter import Language, TimeFormatter


def _text(key: str, language: str | None = None) -> str:
    return get_memory_atom_text(key, language)


def compile_memory_atom(
    atom: MemoryAtom,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a single MemoryAtom into the requested target."""
    alias = atom.get_alias()

    if target == MemoryCompileTarget.PROMPT_FULL:
        text = _render_full_context(
            atom,
            max_content_length=options.max_content_length,
            stale_days=options.stale_days,
            language=options.language,
        )
    elif target == MemoryCompileTarget.PROMPT_INDEX:
        text = _render_index_context(
            atom,
            max_summary_length=options.max_summary_length,
            stale_days=options.stale_days,
            language=options.language,
        )
    elif target == MemoryCompileTarget.DENSE_EMBEDDING:
        text = _render_dense_embedding(atom)
    elif target == MemoryCompileTarget.SPARSE_EMBEDDING:
        text = _render_sparse_embedding(atom)
    elif target == MemoryCompileTarget.AGENT_PROFILE_MENU:
        text = _render_agent_profile(atom, language=options.language)
    elif target == MemoryCompileTarget.MTP_READ:
        text = _render_mtp_read(
            atom,
            max_content_length=options.max_content_length,
            stale_days=options.stale_days,
            language=options.language,
        )
    elif target == MemoryCompileTarget.SHARED_CONTEXT:
        text = _render_shared_context(
            atom,
            max_content_length=options.max_content_length,
            stale_days=options.stale_days,
            language=options.language,
        )
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
        metadata={
            "requested_alias": options.requested_alias,
            "canonical_alias": options.canonical_alias or alias,
        },
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
    language: str | None = None,
) -> str:
    content = _truncate_content(memory.payload.content, max_content_length, language)
    confidence_str = _format_confidence(memory, language)
    alias = memory.get_alias()
    tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or _text("memory_tags_empty", language)
    time_str = TimeFormatter(
        language=_time_formatter_language(language),
        stale_days=stale_days,
    ).format(memory.meta.updated_at)

    history = ""
    if memory.payload.history_summary:
        history_lines = [f"\n**{_text('memory_full_change_log_label', language)}:**"]
        history_lines.extend([f"- {item}" for item in memory.payload.history_summary])
        history = "\n".join(history_lines)

    return _text("memory_full_item_template", language).format(
        alias=alias,
        title=memory.index.title,
        type=memory.index.memory_type.value,
        time=time_str,
        confidence=confidence_str,
        tags=tags,
        content_label=_text("memory_full_content_label", language),
        content=content,
        history=history,
    )


def _render_index_context(
    memory: MemoryAtom,
    max_summary_length: int = 100,
    stale_days: int = 90,
    language: str | None = None,
) -> str:
    alias = memory.get_alias()
    confidence_str = _format_confidence(memory, language)
    tags = ", ".join(f"`{tag}`" for tag in memory.index.tags) or _text("memory_tags_empty", language)
    time_str = TimeFormatter(
        language=_time_formatter_language(language),
        stale_days=stale_days,
    ).format(memory.meta.updated_at)

    summary = memory.index.summary
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length] + "..."

    return _text("memory_index_item_template", language).format(
        alias=alias,
        title=memory.index.title,
        type=memory.index.memory_type.value,
        time=time_str,
        confidence=confidence_str,
        tags=tags,
        summary_label=_text("memory_index_summary_label", language),
        summary=summary,
    )


def _render_agent_profile(memory: MemoryAtom, language: str | None = None) -> str:
    alias = memory.get_alias()
    title = memory.index.title if memory.index.title else _text("memory_agent_profile_untitled", language)
    summary = memory.index.summary if memory.index.summary else ""

    return _text("memory_agent_profile_item_template", language).format(
        alias=alias,
        title=title,
        summary=summary,
    )


def _render_mtp_read(
    memory: MemoryAtom,
    max_content_length: int = 500,
    stale_days: int = 90,
    language: str | None = None,
) -> str:
    return _render_full_context(
        memory,
        max_content_length=max_content_length,
        stale_days=stale_days,
        language=language,
    )


def _render_shared_context(
    memory: MemoryAtom,
    max_content_length: int = 500,
    stale_days: int = 90,
    language: str | None = None,
) -> str:
    return _render_full_context(
        memory,
        max_content_length=max_content_length,
        stale_days=stale_days,
        language=language,
    )


def _format_confidence(memory: MemoryAtom, language: str | None = None) -> str:
    score = memory.meta.confidence_score
    status = memory.meta.verification_status

    status_str = ""
    if status == VerificationStatus.VERIFIED:
        status_str = f" {_text('memory_status_verified', language)}"
    elif status == VerificationStatus.DEPRECATED:
        status_str = f" {_text('memory_status_deprecated', language)}"
    elif status == VerificationStatus.HALLUCINATION:
        status_str = f" {_text('memory_status_hallucination', language)}"
    elif score < 0.7:
        status_str = f" {_text('memory_status_unverified', language)}"

    if score >= 0.9:
        return f"{score:.0%} ({_text('memory_confidence_high', language)}){status_str}"
    if score >= 0.7:
        return f"{score:.0%} ({_text('memory_confidence_medium', language)}){status_str}"
    return f"{score:.0%} ({_text('memory_confidence_low', language)}){status_str}"


def _truncate_content(content: str, max_length: int, language: str | None = None) -> str:
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]

    for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_length // 2:
            truncated = truncated[:last_sep + len(sep)]
            break

    return truncated + f"\n\n{_text('memory_truncation_notice', language)}"


def _time_formatter_language(language: str | None = None) -> Language:
    resolved = resolve_language(explicit=language)
    return Language.ENGLISH if resolved == I18nLanguage.EN else Language.CHINESE
