"""MTP-oriented memory target handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.common import (
    build_artifact,
    is_resolve_terminal,
    render_resolve_terminal,
)
from hivememory.engines.memory_compiler.handlers.prompt import _render_full_from_ir
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n.memory_compiler import (
    get_pending_atom_text,
    get_resolve_result_text,
)


def compile_mtp_read(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.status.is_redirect:
        text = _render_redirect_read(unit, options)
    elif unit.identity.source_kind == "atom":
        text = _render_full_from_ir(
            unit,
            options.max_content_length,
            options.stale_days,
            options.language,
        )
    elif unit.identity.source_kind == "pending":
        text = _render_pending_read(unit, options.language)
    elif is_resolve_terminal(unit):
        text = render_resolve_terminal(unit, options)
    else:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, text, options)


def compile_mtp_redirect_notice(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if not unit.status.is_redirect:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    text = _resolve_text("resolve_redirect_run_notice", options.language).format(
        requested_alias=unit.identity.redirected_from or options.requested_alias or "",
        canonical_alias=unit.identity.alias or "",
    )
    return build_artifact(unit, target, text, options)


def _render_redirect_read(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    return _resolve_text("resolve_redirect_read", options.language).format(
        requested_alias=unit.identity.redirected_from or options.requested_alias or "",
        canonical_alias=unit.identity.alias or "",
        content=unit.content.content or "",
    )


def _render_pending_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    status = unit.status

    if not status.is_terminal:
        if status.source_verb == "UPDATE":
            return _render_pending_revision_read(unit, language)
        return _render_pending_draft_read(unit, language)

    if status.is_discarded:
        return _render_pending_discarded_read(unit, language)
    if status.source_state == "failed":
        return _render_pending_failed_read(unit, language)
    if status.source_state == "cancelled":
        return _pending_text("pending_read_cancelled", language).format(
            pending_alias=unit.identity.alias,
        )
    if status.source_state == "settled":
        canonical_alias = unit.metadata.get("canonical_alias") or ""
        return _pending_text("pending_read_settled", language).format(
            pending_alias=unit.identity.alias,
            canonical_line=f"canonical alias: {canonical_alias}\n" if canonical_alias else "",
        )
    return _pending_text("pending_read_expired", language).format(
        pending_alias=unit.identity.alias,
    )


def _render_pending_draft_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    title = unit.content.title or ""
    return _pending_text("pending_read_write", language).format(
        pending_alias=unit.identity.alias,
        status=unit.status.source_state,
        title_line=f"title: {title}\n" if title else "",
        content=unit.content.content or "",
    )


def _render_pending_revision_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    base_alias = unit.metadata.get("base_alias") or ""
    instruction = unit.content.instruction or ""
    content = unit.content.content or ""
    return _pending_text("pending_read_update", language).format(
        pending_alias=unit.identity.alias,
        base_alias=base_alias,
        status=unit.status.source_state,
        instruction_line=f"instruction: {instruction}\n" if instruction else "",
        content=content,
    )


def _render_pending_discarded_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    return _pending_text("pending_read_discarded", language).format(
        pending_alias=unit.identity.alias,
        message_line=f"message: {unit.status.message}\n" if unit.status.message else "",
        reason_line=f"reason: {unit.status.reason}\n" if unit.status.reason else "",
    ).rstrip()


def _render_pending_failed_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    error = unit.status.error or "Memory generation failed."
    return _pending_text("pending_read_failed", language).format(
        pending_alias=unit.identity.alias,
        error=error,
    )


def _pending_text(key: str, language: str | None = None) -> str:
    return get_pending_atom_text(key, language)


def _resolve_text(key: str, language: str | None = None) -> str:
    return get_resolve_result_text(key, language)
