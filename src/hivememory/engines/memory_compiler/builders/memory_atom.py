"""MemoryAtom → MemoryUnitIR builder."""

from __future__ import annotations

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)


def build_memory_atom_ir(atom: MemoryAtom) -> MemoryUnitIR:
    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="atom",
            alias=atom.get_alias(),
            memory_id=str(atom.id),
        ),
        content=MemoryContentIR(
            title=atom.index.title,
            summary=atom.index.summary,
            content=atom.payload.content,
            tags=list(atom.index.tags),
            memory_type=atom.index.memory_type.value,
        ),
        status=MemoryStatusIR(),
        metadata={
            "confidence_score": atom.meta.confidence_score,
            "verification_status": atom.meta.verification_status,
            "updated_at": atom.meta.updated_at,
            # artifact 关闭时的轻量历史 fallback。
            # TODO(history-compiler): 接入统一历史信息编译后，再决定是否继续透传。
            "history_summary": atom.payload.history_summary,
        },
    )
