"""Embedding-oriented memory target handlers."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.common import build_artifact
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)


def compile_dense_embedding(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind != "atom":
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, _render_dense_embedding_from_ir(unit), options)


def compile_sparse_embedding(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind != "atom":
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, _render_sparse_embedding_from_ir(unit), options)


def _render_dense_embedding_from_ir(unit: MemoryUnitIR) -> str:
    return (
        f"Title: {unit.content.title or ''}\n"
        f"Type: {unit.content.memory_type or ''}\n"
        f"Tags: {', '.join(unit.content.tags)}\n"
        f"Summary: {unit.content.summary or ''}"
    )


def _render_sparse_embedding_from_ir(unit: MemoryUnitIR) -> str:
    tags_string = " ".join(unit.content.tags)
    title = unit.content.title or ""
    return (
        f"{title} {title} "
        f"{tags_string} {tags_string} "
        f"{unit.content.summary or ''}"
    )
