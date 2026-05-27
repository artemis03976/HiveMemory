"""Compatibility adapter functions for MemoryAtom compilation."""

from __future__ import annotations

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler.handlers.memory_atom import (
    _render_agent_profile,
    _render_dense_embedding,
    _render_full_context,
    _render_index_context,
    _render_sparse_embedding,
)
from hivememory.engines.memory_compiler.models import MemoryCompileOptions


def compile_atom_full(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return _render_full_context(
        memory=atom,
        max_content_length=options.max_content_length,
        stale_days=options.stale_days,
    )


def compile_atom_index(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return _render_index_context(
        memory=atom,
        max_summary_length=options.max_summary_length,
        stale_days=options.stale_days,
    )


def compile_atom_dense(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return _render_dense_embedding(atom)


def compile_atom_sparse(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return _render_sparse_embedding(atom)


def compile_atom_agent_profile(atom: MemoryAtom, options: MemoryCompileOptions) -> str:
    return _render_agent_profile(atom)
