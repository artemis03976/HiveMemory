"""ResolveResult → MemoryUnitIR builder.

只处理两条需要 IR 的路径：
- redirect: SETTLED + canonical alias/uuid，需携带预取的 canonical atom 内容
- terminal: discarded / failed / expired，空内容 + 状态标记
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.builders.memory_atom import build_memory_atom_ir
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.resolver import ResolveResult

_TERMINAL_KINDS = {"discarded", "failed", "expired"}


def build_resolve_result_ir(resolve_result: "ResolveResult") -> MemoryUnitIR:
    """Build IR for redirect or terminal ResolveResult kinds only."""
    kind = resolve_result.kind
    settlement = resolve_result.settlement

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
            status=MemoryStatusIR(is_redirect=True),
            metadata=inner.metadata,
        )

    # terminal kinds: discarded / failed / expired
    if kind not in _TERMINAL_KINDS:
        raise ValueError(
            f"build_resolve_result_ir only handles 'redirect' and terminal kinds "
            f"({_TERMINAL_KINDS}), got '{kind}'."
        )
    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="resolve_result",
            alias=resolve_result.requested_alias,
        ),
        content=MemoryContentIR(),
        status=MemoryStatusIR(
            is_terminal=True,
            is_discarded=kind == "discarded",
            message=settlement.message if settlement else None,
            reason=settlement.reason if settlement else None,
            error=settlement.error if settlement else None,
        ),
    )