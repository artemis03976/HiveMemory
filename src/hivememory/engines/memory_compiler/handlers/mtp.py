"""MTP-oriented memory target handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.common import (
    build_artifact,
    format_optional_field,
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


def _render_redirect_read(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    return _resolve_text("resolve_redirect_read", options.language).format(
        requested_alias=format_optional_field(
            unit.identity.redirected_from or options.requested_alias,
            options.language,
        ),
        canonical_alias=format_optional_field(unit.identity.alias, options.language),
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
        return _pending_text("pending_read_settled", language).format(
            pending_alias=unit.identity.alias,
            canonical_alias=format_optional_field(
                unit.metadata.get("canonical_alias"),
                language,
            ),
        )
    return _pending_text("pending_read_expired", language).format(
        pending_alias=unit.identity.alias,
    )


def _render_pending_draft_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    return _pending_text("pending_read_write", language).format(
        pending_alias=unit.identity.alias,
        status=unit.status.source_state,
        title=format_optional_field(unit.content.title, language),
        content=unit.content.content or "",
    )


def _render_pending_revision_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    content = unit.content.content or ""
    return _pending_text("pending_read_update", language).format(
        pending_alias=unit.identity.alias,
        base_alias=format_optional_field(unit.metadata.get("base_alias"), language),
        status=unit.status.source_state,
        instruction=format_optional_field(unit.content.instruction, language),
        content=content,
    )


def _render_pending_discarded_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    return _pending_text("pending_read_discarded", language).format(
        pending_alias=unit.identity.alias,
        message=format_optional_field(unit.status.message, language),
        reason=format_optional_field(unit.status.reason, language),
    ).rstrip()


def _render_pending_failed_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    return _pending_text("pending_read_failed", language).format(
        pending_alias=unit.identity.alias,
        error=format_optional_field(unit.status.error, language),
    )


def _pending_text(key: str, language: str | None = None) -> str:
    return get_pending_atom_text(key, language)


def _resolve_text(key: str, language: str | None = None) -> str:
    return get_resolve_result_text(key, language)
