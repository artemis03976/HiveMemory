"""MemoryCompiler unified memory-to-text compilation layer."""

from __future__ import annotations

from hivememory.engines.memory_compiler.compiler import MemoryCompiler
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR, MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CascadeStrategyConfig,
    CompiledMemory,
    CompiledMemoryArtifact,
    CompiledMemoryEnvelope,
    CompactStrategyConfig,
    FullStrategyConfig,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
    RetrievalStrategyConfig,
)

__all__ = [
    "MemoryCompiler",
    "MemoryCompileTarget",
    "MemoryEnvelopeTarget",
    "CompiledMemory",
    "CompiledMemoryArtifact",
    "CompiledMemoryEnvelope",
    "MemoryEnvelopeSection",
    "MemoryCompileOptions",
    "MemorySectionIR",
    "MemoryBundleIR",
    "MemoryUnitIR",
    "FullStrategyConfig",
    "CascadeStrategyConfig",
    "CompactStrategyConfig",
    "RetrievalStrategyConfig",
]
