"""PendingAtom rendering helpers for target-first handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.i18n.memory_compiler import get_pending_atom_text


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
    if status.source_state == "settled":
        canonical_alias = unit.metadata.get("canonical_alias") or ""
        return _t("pending_read_settled", language).format(
            pending_alias=unit.identity.alias,
            canonical_line=f"canonical alias: {canonical_alias}\n" if canonical_alias else "",
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


def _t(key: str, language: str | None = None) -> str:
    return get_pending_atom_text(key, language)
