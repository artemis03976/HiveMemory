"""薄适配器层 — 委托给现有 MemoryAtomRenderer。"""

from __future__ import annotations

from hivememory.core.models import MemoryAtom
from hivememory.utils.memory_atom_renderer import MemoryAtomRenderer
from hivememory.engines.memory_compiler.models import MemoryCompileOptions


# ========== MemoryAtom 适配器 ==========


def compile_atom_full(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return MemoryAtomRenderer.for_full_context(
        memory=atom,
        max_content_length=options.max_content_length,
        stale_days=options.stale_days,
    )


def compile_atom_index(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return MemoryAtomRenderer.for_index_context(
        memory=atom,
        max_summary_length=options.max_summary_length,
        stale_days=options.stale_days,
    )


def compile_atom_dense(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return MemoryAtomRenderer.for_dense_embedding(atom)


def compile_atom_sparse(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return MemoryAtomRenderer.for_sparse_embedding(atom)


def compile_atom_agent_profile(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return MemoryAtomRenderer.for_agent_profile(atom)
