"""PendingAtom compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)

if TYPE_CHECKING:
    from hivememory.core.models import MemoryAtom
    from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement


def compile_pending_atom(
    pending: "PendingAtom",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a single PendingAtom into the requested target."""
    if target == MemoryCompileTarget.MTP_READ:
        text = _render_read(pending)
    elif target == MemoryCompileTarget.SHARED_CONTEXT:
        text = _render_read(pending)
    elif target == MemoryCompileTarget.MTP_ACK:
        text = _render_ack(pending)
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


def _render_read(pending: "PendingAtom") -> str:
    if pending.source_verb == "UPDATE":
        return _render_revision_read(pending)
    return _render_draft_read(pending)


def _render_draft_read(pending: "PendingAtom") -> str:
    from hivememory.engines.generation.models import WriteFocus

    focus = pending.focus
    if not isinstance(focus, WriteFocus):
        raise TypeError("WRITE pending atom must carry WriteFocus.")

    lines = [f"[{pending.pending_alias}] (runtime pending atom):"]
    lines.append("status: pending")
    lines.append("source: WRITE")
    if focus.title:
        lines.append(f"title: {focus.title}")
    lines.append("")
    lines.append("content:")
    lines.append(focus.content)
    lines.append("")
    lines.append(
        "note: This is a runtime pending atom. "
        "Final memory generation is asynchronous."
    )
    return "\n".join(lines)


def _render_revision_read(pending: "PendingAtom") -> str:
    from hivememory.engines.generation.models import UpdateFocus

    focus = pending.focus
    if not isinstance(focus, UpdateFocus):
        raise TypeError("UPDATE pending atom must carry UpdateFocus.")

    lines = [
        f"[{pending.pending_alias}] "
        f"(pending revision of '{focus.base_alias}'):"
    ]
    lines.append("status: revision")
    if focus.instruction:
        lines.append(f"instruction: {focus.instruction}")
    lines.append("")
    lines.append("new content:")
    lines.append(focus.content or "")
    lines.append("")
    lines.append(
        "note: This is a pending revision. "
        "The original memory has not been modified yet."
    )
    return "\n".join(lines)


def _render_ack(pending: "PendingAtom") -> str:
    from hivememory.engines.generation.models import UpdateFocus

    if pending.source_verb == "UPDATE":
        focus = pending.focus
        if not isinstance(focus, UpdateFocus):
            raise TypeError("UPDATE pending atom must carry UpdateFocus.")
        return (
            f"Memory '{focus.base_alias}' update accepted as "
            f"pending revision '{pending.pending_alias}'.\n"
            f"It is readable during this run via READ. "
            f"Final memory update will complete asynchronously."
        )
    return (
        f"Memory accepted as pending atom '{pending.pending_alias}'.\n"
        f"It is readable during this run via READ. "
        f"Final memory generation will complete asynchronously."
    )


def _render_redirect_read(
    *,
    requested_alias: str,
    canonical_alias: str,
    atom: "MemoryAtom",
    settlement: "PendingAtomSettlement | None" = None,
) -> str:
    status = settlement.resolution.value if settlement else "redirected"
    lines = [
        "[Alias Redirected]",
        f"Requested alias: {requested_alias}",
        f"Canonical alias: {canonical_alias}",
        f"Status: {status}",
        "",
        f"[{canonical_alias}]:",
        atom.payload.content,
        "",
        f"Action: Use '{canonical_alias}' for future READ/RUN/UPDATE calls.",
    ]
    return "\n".join(lines)


def _render_redirect_run_notice(
    *,
    requested_alias: str,
    canonical_alias: str,
    settlement: "PendingAtomSettlement | None" = None,
) -> str:
    status = settlement.resolution.value if settlement else "redirected"
    return "\n".join(
        [
            "[Alias Redirected]",
            f"Requested alias: {requested_alias}",
            f"Canonical alias: {canonical_alias}",
            f"Status: {status}",
            f"Action: Use '{canonical_alias}' for future RUN calls.",
            "",
        ]
    )


def _render_settled_without_atom(
    *,
    requested_alias: str,
    settlement: "PendingAtomSettlement | None",
) -> str:
    status = settlement.resolution.value if settlement else "settled"
    message = settlement.message if settlement and settlement.message else ""
    reason = settlement.reason if settlement and settlement.reason else ""
    lines = [
        f"[{requested_alias}]",
        f"status: {status}",
        "materialized: false",
    ]
    if message:
        lines.append(f"message: {message}")
    if reason:
        lines.append(f"reason: {reason}")
    lines.append("")
    lines.append("Action: Use SEARCH to locate related finalized memory if needed.")
    return "\n".join(lines)
