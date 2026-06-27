"""MemoryCompiler unified memory-to-artifact compilation layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from hivememory.core.models import MemoryAtom

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

def compile_retrieval_context(
    memories: "List[MemoryAtom]",
    options: "Optional[MemoryCompileOptions]" = None,
) -> str:
    """
    Compile a list of MemoryAtom objects into a retrieval context string.

    Separates AGENT_PROFILE atoms into a dedicated section. Returns "" for
    empty input. Defaults to FullStrategyConfig when no strategy is specified.
    """
    if not memories:
        return ""

    from hivememory.core.models import MemoryType
    from hivememory.engines.memory_compiler.builders import build_memory_atom_ir

    opts = options or MemoryCompileOptions(retrieval_strategy_config=FullStrategyConfig())
    if opts.retrieval_strategy_config is None:
        opts = opts.model_copy(update={"retrieval_strategy_config": FullStrategyConfig()})

    regular = [m for m in memories if getattr(getattr(m, "index", None), "memory_type", None) != MemoryType.AGENT_PROFILE]
    agents  = [m for m in memories if getattr(getattr(m, "index", None), "memory_type", None) == MemoryType.AGENT_PROFILE]

    compiler = MemoryCompiler()
    sections = [
        MemorySectionIR(kind="memories", units=[build_memory_atom_ir(m) for m in regular]),
        MemorySectionIR(
            kind="agent_profiles",
            artifacts=[compiler.compile(a, MemoryCompileTarget.AGENT_PROFILE_MENU) for a in agents],
        ),
    ]
    return compiler.compile(sections, MemoryEnvelopeTarget.RETRIEVAL_CONTEXT, opts).text


__all__ = [
    "compile_retrieval_context",
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
