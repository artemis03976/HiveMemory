"""PendingAtom → MemoryUnitIR 构建器。"""

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


def build_pending_atom_ir(pending: PendingAtom) -> MemoryUnitIR:
    """只编译在途的写入意图：终态句柄由读取器转为引用解析结果交付，不按意图编译。"""
    from hivememory.core.models.pending import UpdateFocus

    if not pending.status.is_in_flight:
        raise ValueError(
            f"只编译在途的写入意图，收到 {pending.status.value}: {pending.pending_alias}"
        )
    focus = pending.focus
    if isinstance(focus, UpdateFocus):
        content_ir = MemoryContentIR(instruction=focus.instruction, content=focus.content)
        base_alias: str | None = focus.base_alias
    else:
        content_ir = MemoryContentIR(title=focus.title, content=focus.content)
        base_alias = None

    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="pending",
            alias=pending.pending_alias,
        ),
        content=content_ir,
        status=MemoryStatusIR(
            source_state=pending.status.value,
            source_verb=pending.source_verb,
        ),
        metadata={"base_alias": base_alias},
    )
