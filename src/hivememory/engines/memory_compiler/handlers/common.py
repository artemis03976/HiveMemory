"""Shared helpers for target-first memory compiler handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.resolve_result import (
    _t as _resolve_text,
)
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
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
    if is_resolve_terminal(unit) and unit.status.error is not None:
        return "failed"
    if is_resolve_terminal(unit):
        return "expired"
    return unit.status.source_state


def is_resolve_terminal(unit: MemoryUnitIR) -> bool:
    return unit.identity.source_kind == "resolve_result" and unit.status.is_terminal


def render_resolve_terminal(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    status = unit.status
    alias = unit.identity.alias or options.requested_alias or ""

    if status.is_discarded:
        return _resolve_text("resolve_discarded", options.language).format(
            requested_alias=alias,
            message_line=f"message: {status.message}\n" if status.message else "",
            reason_line=f"reason: {status.reason}\n" if status.reason else "",
        ).rstrip()

    if status.error is not None:
        return _resolve_text("resolve_failed", options.language).format(
            requested_alias=alias,
            error_line=f"error: {status.error}\n",
            message_line=f"message: {status.message}\n" if status.message else "",
            reason_line=f"reason: {status.reason}\n" if status.reason else "",
        ).rstrip()

    return _resolve_text("resolve_expired", options.language).format(requested_alias=alias)
