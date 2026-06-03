"""PendingAtom compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n.memory_compiler import get_pending_atom_text

if TYPE_CHECKING:
    from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement


def compile_pending_atom(
    pending: "PendingAtom",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a single PendingAtom into the requested target."""
    from hivememory.core.models.pending import PendingAtomStatus

    if target == MemoryCompileTarget.MTP_ACK:
        text = _render_ack(pending, options.language)
    elif target in (MemoryCompileTarget.MTP_READ, MemoryCompileTarget.SHARED_CONTEXT):
        status = pending.status
        if status.is_in_flight:
            text = _render_read(pending, options.language)
        elif status == PendingAtomStatus.SETTLED:
            text = _render_settled_read(pending, options.language)
        elif status == PendingAtomStatus.FAILED:
            text = _render_failed_read(pending, options.language)
        elif status == PendingAtomStatus.CANCELLED:
            text = _render_cancelled_read(pending, options.language)
        else:  # EXPIRED
            text = _render_expired_read(pending, options.language)
    else:
        raise ValueError(f"Unsupported target '{target}' for PendingAtom source.")

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind="pending",
        alias=pending.pending_alias,
        status=pending.status.value,
        metadata={
            "requested_alias": options.requested_alias,
            "canonical_alias": options.canonical_alias,
        },
    )


def _render_read(pending: "PendingAtom", language: str | None = None) -> str:
    if pending.source_verb == "UPDATE":
        return _render_revision_read(pending, language)
    return _render_draft_read(pending, language)


def _render_draft_read(pending: "PendingAtom", language: str | None = None) -> str:
    from hivememory.core.models import WriteFocus

    focus = pending.focus
    if not isinstance(focus, WriteFocus):
        raise TypeError("WRITE pending atom must carry WriteFocus.")

    return get_pending_atom_text("pending_read_write", language).format(
        pending_alias=pending.pending_alias,
        status=pending.status.value,
        title_line=f"title: {focus.title}\n" if focus.title else "",
        content=focus.content,
    )


def _render_revision_read(pending: "PendingAtom", language: str | None = None) -> str:
    from hivememory.core.models import UpdateFocus

    focus = pending.focus
    if not isinstance(focus, UpdateFocus):
        raise TypeError("UPDATE pending atom must carry UpdateFocus.")

    return get_pending_atom_text("pending_read_update", language).format(
        pending_alias=pending.pending_alias,
        base_alias=focus.base_alias,
        status=pending.status.value,
        instruction_line=(
            f"instruction: {focus.instruction}\n" if focus.instruction else ""
        ),
        content=focus.content or "",
    )


def _render_ack(pending: "PendingAtom", language: str | None = None) -> str:
    from hivememory.core.models import UpdateFocus

    if pending.source_verb == "UPDATE":
        focus = pending.focus
        if not isinstance(focus, UpdateFocus):
            raise TypeError("UPDATE pending atom must carry UpdateFocus.")
        return get_pending_atom_text("pending_ack_update", language).format(
            base_alias=focus.base_alias,
            pending_alias=pending.pending_alias,
        )
    return get_pending_atom_text("pending_ack_write", language).format(
        pending_alias=pending.pending_alias,
    )


def _render_settled_read(pending: "PendingAtom", language: str | None = None) -> str:
    settlement = pending.settlement
    canonical_alias = settlement.canonical_alias if settlement else ""
    message = settlement.message if settlement and settlement.message else ""
    return get_pending_atom_text("pending_read_settled", language).format(
        pending_alias=pending.pending_alias,
        resolution=settlement.resolution.value if settlement else "settled",
        canonical_line=(
            f"canonical alias: {canonical_alias}\n" if canonical_alias else ""
        ),
        message_line=f"message: {message}\n" if message else "",
        action_line=(
            f"Action: Use '{canonical_alias}' for future READ/UPDATE calls."
            if canonical_alias else ""
        ),
    ).rstrip()


def _render_failed_read(pending: "PendingAtom", language: str | None = None) -> str:
    settlement = pending.settlement
    error = settlement.error if settlement else None
    return get_pending_atom_text("pending_read_failed", language).format(
        pending_alias=pending.pending_alias,
        error=error or "Memory generation failed.",
    )


def _render_cancelled_read(pending: "PendingAtom", language: str | None = None) -> str:
    return get_pending_atom_text("pending_read_cancelled", language).format(
        pending_alias=pending.pending_alias,
    )


def _render_expired_read(pending: "PendingAtom", language: str | None = None) -> str:
    return get_pending_atom_text("pending_read_expired", language).format(
        pending_alias=pending.pending_alias,
    )
