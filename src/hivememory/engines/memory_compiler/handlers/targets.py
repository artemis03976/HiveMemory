"""Target-first MemoryUnitIR compilation dispatch."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.agent_profile import (
    compile_agent_profile_menu,
)
from hivememory.engines.memory_compiler.handlers.embedding import (
    compile_dense_embedding,
    compile_sparse_embedding,
)
from hivememory.engines.memory_compiler.handlers.mtp import (
    compile_mtp_read,
    compile_mtp_redirect_notice,
)
from hivememory.engines.memory_compiler.handlers.prompt import (
    compile_prompt_full,
    compile_prompt_index,
    compile_shared_context,
)
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)


def compile_from_ir(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a normalized memory unit into the requested target."""
    if target == MemoryCompileTarget.PROMPT_FULL:
        return compile_prompt_full(unit, target, options)
    if target == MemoryCompileTarget.PROMPT_INDEX:
        return compile_prompt_index(unit, target, options)
    if target == MemoryCompileTarget.MTP_READ:
        return compile_mtp_read(unit, target, options)
    if target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
        return compile_mtp_redirect_notice(unit, target, options)
    if target == MemoryCompileTarget.SHARED_CONTEXT:
        return compile_shared_context(unit, target, options)
    if target == MemoryCompileTarget.DENSE_EMBEDDING:
        return compile_dense_embedding(unit, target, options)
    if target == MemoryCompileTarget.SPARSE_EMBEDDING:
        return compile_sparse_embedding(unit, target, options)
    if target == MemoryCompileTarget.AGENT_PROFILE_MENU:
        return compile_agent_profile_menu(unit, target, options)
    if target == MemoryCompileTarget.RUNNABLE_TOOL:
        raise ValueError("RUNNABLE_TOOL target is reserved for Phase 3.")

    raise ValueError(f"Unsupported target '{target}'.")
