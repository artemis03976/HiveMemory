"""PendingAtom → MemoryUnitIR builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)

if TYPE_CHECKING:
    from hivememory.core.models.pending import PendingAtom

_TERMINAL_STATUSES = {"settled", "failed", "cancelled", "expired"}


def build_pending_atom_ir(pending: "PendingAtom") -> MemoryUnitIR:
    from hivememory.core.models.pending import UpdateFocus, WriteFocus

    focus = pending.focus
    status_val = pending.status.value
    settlement = pending.settlement

    if isinstance(focus, WriteFocus):
        content_ir = MemoryContentIR(
            title=focus.title,
            content=focus.content,
        )
    else:  # UpdateFocus
        assert isinstance(focus, UpdateFocus)
        content_ir = MemoryContentIR(
            instruction=focus.instruction,
            content=focus.content,
        )

    status_ir = MemoryStatusIR(
        source_state=status_val,
        source_verb=pending.source_verb,
        is_terminal=status_val in _TERMINAL_STATUSES,
        is_discarded=(
            status_val == "settled"
            and settlement is not None
            and settlement.resolution is not None
            and settlement.resolution.value == "discarded"
        ),
        message=settlement.message if settlement else None,
        error=settlement.error if settlement else None,
        reason=settlement.reason if settlement else None,
    )

    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="pending",
            alias=pending.pending_alias,
        ),
        content=content_ir,
        status=status_ir,
        metadata={
            "base_alias": focus.base_alias if isinstance(focus, UpdateFocus) else None,
            "canonical_alias": settlement.canonical_alias if settlement else None,
        },
    )
