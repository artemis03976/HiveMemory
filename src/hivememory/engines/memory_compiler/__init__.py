"""MemoryCompiler unified memory-to-artifact compilation layer."""

from hivememory.engines.memory_compiler.compiler import MemoryCompiler
from hivememory.engines.memory_compiler.models import (
    CascadeStrategyConfig,
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
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR, MemoryUnitIR

__all__ = [
    "MemoryCompiler",
    "MemoryCompileTarget",
    "MemoryEnvelopeTarget",
    "CompiledMemoryArtifact",
    "CompiledMemoryEnvelope",
    "MemoryEnvelopeSection",
    "MemoryCompileOptions",
    "MemorySectionIR",
    "MemoryBundleIR",
    "MemoryUnitIR",
    # Phase A: retrieval strategy configs
    "FullStrategyConfig",
    "CascadeStrategyConfig",
    "CompactStrategyConfig",
    "RetrievalStrategyConfig",
]
