"""ResolveResult → MemoryUnitIR builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.builders.memory_atom import build_memory_atom_ir
from hivememory.engines.memory_compiler.builders.pending_atom import build_pending_atom_ir
from hivememory.engines.memory_compiler.ir import (
    MemoryIdentityIR,
    MemoryContentIR,
    MemoryStatusIR,
    MemoryUnitIR,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.resolver import ResolveResult

_TERMINAL_KINDS = {"discarded", "failed", "expired"}


def build_resolve_result_ir(resolve_result: "ResolveResult") -> MemoryUnitIR:
    kind = resolve_result.kind
    settlement = resolve_result.settlement

    # redirect: fill identity with both aliases, delegate content to atom builder
    if kind == "redirect" and resolve_result.atom is not None:
        inner = build_memory_atom_ir(resolve_result.atom)
        return MemoryUnitIR(
            identity=MemoryIdentityIR(
                source_kind="resolve_result",
                alias=resolve_result.canonical_alias or resolve_result.atom.get_alias(),
                redirected_from=resolve_result.requested_alias,
                memory_id=str(resolve_result.atom.id),
            ),
            content=inner.content,
            status=MemoryStatusIR(
                resolve_state=kind,
                settlement_state=settlement.resolution.value if settlement else None,
                is_terminal=False,
            ),
            metadata=inner.metadata,
        )

    # pending: delegate entirely to pending builder
    if kind == "pending" and resolve_result.pending is not None:
        inner = build_pending_atom_ir(resolve_result.pending)
        return MemoryUnitIR(
            identity=MemoryIdentityIR(
                source_kind="resolve_result",
                alias=inner.identity.alias,
            ),
            content=inner.content,
            status=MemoryStatusIR(
                resolve_state=kind,
                source_state=inner.status.source_state,
                source_verb=inner.status.source_verb,
                is_terminal=inner.status.is_terminal,
                settlement_state=inner.status.settlement_state,
                message=inner.status.message,
                error=inner.status.error,
                reason=inner.status.reason,
            ),
            metadata=inner.metadata,
        )

    # atom: delegate entirely to atom builder
    if kind == "atom" and resolve_result.atom is not None:
        inner = build_memory_atom_ir(resolve_result.atom)
        return MemoryUnitIR(
            identity=MemoryIdentityIR(
                source_kind="resolve_result",
                alias=inner.identity.alias,
                memory_id=inner.identity.memory_id,
            ),
            content=inner.content,
            status=MemoryStatusIR(resolve_state=kind),
            metadata=inner.metadata,
        )

    # terminal kinds: discarded / failed / expired
    error = settlement.error if settlement else None
    message = settlement.message if settlement else None
    reason = settlement.reason if settlement else None
    settlement_state = settlement.resolution.value if settlement else None

    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="resolve_result",
            alias=resolve_result.requested_alias,
        ),
        content=MemoryContentIR(),
        status=MemoryStatusIR(
            resolve_state=kind,
            settlement_state=settlement_state,
            is_terminal=kind in _TERMINAL_KINDS,
            error=error,
            message=message,
            reason=reason,
        ),
    )
