"""Prompt-oriented memory target handlers."""

from __future__ import annotations

from hivememory.core.models import VerificationStatus
from hivememory.engines.memory_compiler.handlers.common import (
    build_artifact,
    is_resolve_terminal,
    render_resolve_terminal,
)
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n import get_memory_atom_text
from hivememory.utils.time_formatter import TimeFormatter


def _text(key: str, language: str | None = None) -> str:
    return get_memory_atom_text(key, language)


def compile_prompt_full(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind == "atom" or unit.status.is_redirect:
        text = _render_full_from_ir(
            unit,
            options.max_content_length,
            options.stale_days,
            options.language,
        )
    elif is_resolve_terminal(unit):
        text = render_resolve_terminal(unit, options)
    else:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, text, options)


def compile_prompt_index(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind == "atom" or unit.status.is_redirect:
        text = _render_index_from_ir(
            unit,
            options.max_summary_length,
            options.stale_days,
            options.language,
        )
    elif is_resolve_terminal(unit):
        text = render_resolve_terminal(unit, options)
    else:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, text, options)


def _render_full_from_ir(
    unit: MemoryUnitIR,
    max_content_length: int = 500,
    stale_days: int = 90,
    language: str | None = None,
) -> str:
    content = _truncate_content(unit.content.content or "", max_content_length, language)
    confidence_str = _format_confidence_from_ir(unit, language)
    alias = unit.identity.alias or ""
    tags = ", ".join(f"`{tag}`" for tag in unit.content.tags) or _text("memory_tags_empty", language)
    updated_at = unit.metadata.get("updated_at")
    time_str = (
        TimeFormatter(language=language, stale_days=stale_days).format(updated_at)
        if updated_at is not None
        else _text("memory_time_unknown", language)
    )

    history = ""
    history_summary = unit.metadata.get("history_summary", [])
    if history_summary:
        history_lines = [f"\n**{_text('memory_full_change_log_label', language)}:**"]
        history_lines.extend(f"- {item}" for item in history_summary)
        history = "\n".join(history_lines)

    return _text("memory_full_item_template", language).format(
        alias=alias,
        title=unit.content.title or "",
        type=unit.content.memory_type or "",
        time=time_str,
        confidence=confidence_str,
        tags=tags,
        content_label=_text("memory_full_content_label", language),
        content=content,
        history=history,
    )


def _render_index_from_ir(
    unit: MemoryUnitIR,
    max_summary_length: int = 100,
    stale_days: int = 90,
    language: str | None = None,
) -> str:
    alias = unit.identity.alias or ""
    confidence_str = _format_confidence_from_ir(unit, language)
    tags = ", ".join(f"`{tag}`" for tag in unit.content.tags) or _text("memory_tags_empty", language)
    updated_at = unit.metadata.get("updated_at")
    time_str = (
        TimeFormatter(language=language, stale_days=stale_days).format(updated_at)
        if updated_at is not None
        else _text("memory_time_unknown", language)
    )

    summary = unit.content.summary or ""
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length] + "..."

    return _text("memory_index_item_template", language).format(
        alias=alias,
        title=unit.content.title or "",
        type=unit.content.memory_type or "",
        time=time_str,
        confidence=confidence_str,
        tags=tags,
        summary_label=_text("memory_index_summary_label", language),
        summary=summary,
    )


def _format_confidence_from_ir(unit: MemoryUnitIR, language: str | None = None) -> str:
    score: float = unit.metadata.get("confidence_score", 0.0)
    status = unit.metadata.get("verification_status")

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
