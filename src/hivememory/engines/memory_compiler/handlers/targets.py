"""Target-first MemoryUnitIR compilation dispatch."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.agent_profile import (
    compile_agent_profile_menu,
)
from hivememory.engines.memory_compiler.handlers.common import (
    build_artifact,
    is_resolve_terminal,
    render_resolve_terminal,
)
from hivememory.engines.memory_compiler.handlers.embedding import (
    compile_dense_embedding,
    compile_sparse_embedding,
)
from hivememory.engines.memory_compiler.handlers.pending_atom import (
    _render_read as _render_pending_read,
)
from hivememory.engines.memory_compiler.handlers.prompt import (
    compile_prompt_full,
    compile_prompt_index,
    _render_full_from_ir,
)
from hivememory.engines.memory_compiler.handlers.resolve_result import (
    _t as _resolve_text,
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
        return _compile_mtp_read(unit, target, options)
    if target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
        return _compile_mtp_redirect_notice(unit, target, options)
    if target == MemoryCompileTarget.SHARED_CONTEXT:
        return _compile_shared_context(unit, target, options)
    if target == MemoryCompileTarget.DENSE_EMBEDDING:
        return compile_dense_embedding(unit, target, options)
    if target == MemoryCompileTarget.SPARSE_EMBEDDING:
        return compile_sparse_embedding(unit, target, options)
    if target == MemoryCompileTarget.AGENT_PROFILE_MENU:
        return compile_agent_profile_menu(unit, target, options)
    if target == MemoryCompileTarget.RUNNABLE_TOOL:
        raise ValueError("RUNNABLE_TOOL target is reserved for Phase 3.")

    raise ValueError(f"Unsupported target '{target}'.")


def _compile_mtp_read(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.status.is_redirect:
        text = _render_redirect_read(unit, options)
    elif unit.identity.source_kind == "atom":
        text = _render_full_from_ir(
            unit,
            options.max_content_length,
            options.stale_days,
            options.language,
        )
    elif unit.identity.source_kind == "pending":
        text = _render_pending_read(unit, options.language)
    elif is_resolve_terminal(unit):
        text = render_resolve_terminal(unit, options)
    else:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, text, options)


def _compile_mtp_redirect_notice(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if not unit.status.is_redirect:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    text = _resolve_text("resolve_redirect_run_notice", options.language).format(
        requested_alias=unit.identity.redirected_from or options.requested_alias or "",
        canonical_alias=unit.identity.alias or "",
    )
    return build_artifact(unit, target, text, options)


def _compile_shared_context(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind == "atom" or unit.status.is_redirect:
        text = _render_full_from_ir(
            unit,
            options.max_content_length,
            options.stale_days,
            options.language,
        )
    elif unit.identity.source_kind == "pending":
        text = _render_pending_read(unit, options.language)
    elif is_resolve_terminal(unit):
        text = render_resolve_terminal(unit, options)
    else:
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(unit, target, text, options)


def _render_redirect_read(unit: MemoryUnitIR, options: MemoryCompileOptions) -> str:
    return _resolve_text("resolve_redirect_read", options.language).format(
        requested_alias=unit.identity.redirected_from or options.requested_alias or "",
        canonical_alias=unit.identity.alias or "",
        content=unit.content.content or "",
    )
