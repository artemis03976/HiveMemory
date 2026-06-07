"""Shared helpers for target-first memory compiler handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n.memory_compiler import (
    get_memory_atom_text,
    get_resolve_result_text,
)


def build_artifact(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    text: str,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    metadata = {}
    if unit.identity.source_kind == "atom":
        metadata = {
            "requested_alias": options.requested_alias,
            "canonical_alias": options.canonical_alias or unit.identity.alias,
        }
    elif unit.identity.source_kind == "pending":
        metadata = {
            "requested_alias": options.requested_alias,
            "canonical_alias": options.canonical_alias,
        }

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind=unit.identity.source_kind,
        alias=unit.identity.alias,
        memory_id=unit.identity.memory_id,
        status=artifact_status(unit),
        metadata=metadata,
    )


def artifact_status(unit: MemoryUnitIR) -> str | None:
    if unit.status.is_redirect:
        return "redirect"
    if unit.status.is_discarded:
        return "discarded"
    if is_resolve_terminal(unit):
        return "failed" if unit.status.error is not None else "expired"
    return unit.status.source_state


def is_resolve_terminal(unit: MemoryUnitIR) -> bool:
    return unit.identity.source_kind == "resolve_result" and unit.status.is_terminal


def render_resolve_terminal(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    status = unit.status
    alias = format_optional_field(unit.identity.alias or options.requested_alias, options.language)

    if status.is_discarded:
        return _resolve_text("resolve_discarded", options.language).format(
            requested_alias=alias,
            message=format_optional_field(status.message, options.language),
            reason=format_optional_field(status.reason, options.language),
        ).rstrip()

    if status.error is not None:
        return _resolve_text("resolve_failed", options.language).format(
            requested_alias=alias,
            error=format_optional_field(status.error, options.language),
            message=format_optional_field(status.message, options.language),
            reason=format_optional_field(status.reason, options.language),
        ).rstrip()

    return _resolve_text("resolve_expired", options.language).format(requested_alias=alias)


def _resolve_text(key: str, language: str | None = None) -> str:
    return get_resolve_result_text(key, language)


def format_optional_field(value: object, language: str | None = None) -> str:
    if value is None:
        return get_memory_atom_text("memory_field_empty", language)
    text = str(value)
    return text if text else get_memory_atom_text("memory_field_empty", language)
