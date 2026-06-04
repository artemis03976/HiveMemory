"""PendingAtom compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.builders import build_pending_atom_ir
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n.memory_compiler import get_pending_atom_text

if TYPE_CHECKING:
    from hivememory.core.models.pending import PendingAtom


def compile_pending_atom(
    pending: "PendingAtom",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    unit = build_pending_atom_ir(pending)

    if target == MemoryCompileTarget.MTP_ACK:
        text = _render_ack(unit, options.language)
    elif target in (MemoryCompileTarget.MTP_READ, MemoryCompileTarget.SHARED_CONTEXT):
        text = _render_read(unit, options.language)
    else:
        raise ValueError(f"Unsupported target '{target}' for PendingAtom source.")

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind="pending",
        alias=unit.identity.alias,
        status=unit.status.source_state,
        metadata={
            "requested_alias": options.requested_alias,
            "canonical_alias": options.canonical_alias,
        },
    )


def _render_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    status = unit.status

    # in-flight: PENDING 或 MATERIALIZING，渲染本体文本
    if not status.is_terminal:
        if status.source_verb == "UPDATE":
            return _render_revision_read(unit, language)
        return _render_draft_read(unit, language)

    # discarded 状态
    if status.is_discarded:
        return _render_discarded_read(unit, language)
    if status.source_state == "failed":
        return _render_failed_read(unit, language)
    if status.source_state == "cancelled":
        return _t("pending_read_cancelled", language).format(
            pending_alias=unit.identity.alias,
        )
    # expired 状态
    return _t("pending_read_expired", language).format(
        pending_alias=unit.identity.alias,
        )


def _render_draft_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    title = unit.content.title or ""
    return _t("pending_read_write", language).format(
        pending_alias=unit.identity.alias,
        status=unit.status.source_state,
        title_line=f"title: {title}\n" if title else "",
        content=unit.content.content or "",
    )


def _render_revision_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    base_alias = unit.metadata.get("base_alias") or ""
    instruction = unit.content.instruction or ""
    content = unit.content.content or ""
    return _t("pending_read_update", language).format(
        pending_alias=unit.identity.alias,
        base_alias=base_alias,
        status=unit.status.source_state,
        instruction_line=f"instruction: {instruction}\n" if instruction else "",
        content=content,
    )


def _render_discarded_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    return _t("pending_read_discarded", language).format(
        pending_alias=unit.identity.alias,
        message_line=f"message: {unit.status.message}\n" if unit.status.message else "",
        reason_line=f"reason: {unit.status.reason}\n" if unit.status.reason else "",
    ).rstrip()


def _render_failed_read(unit: MemoryUnitIR, language: str | None = None) -> str:
    error = unit.status.error or "Memory generation failed."
    return _t("pending_read_failed", language).format(
        pending_alias=unit.identity.alias,
        error=error,
    )


def _render_ack(unit: MemoryUnitIR, language: str | None = None) -> str:
    if unit.status.source_verb == "UPDATE":
        base_alias = unit.metadata.get("base_alias") or ""
        return _t("pending_ack_update", language).format(
            base_alias=base_alias,
            pending_alias=unit.identity.alias,
        )
    return _t("pending_ack_write", language).format(
        pending_alias=unit.identity.alias,
    )


def _t(key: str, language: str | None = None) -> str:
    return get_pending_atom_text(key, language)
