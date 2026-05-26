"""按 (source_type, target) 分发编译逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler import adapters
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.models import PendingAtom
    from hivememory.alice.runtime.resolver import ResolveResult


def compile_memory_atom(
    atom: MemoryAtom,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """编译单个 MemoryAtom。"""
    alias = atom.get_alias()
    effective_alias = options.requested_alias or alias

    if target == MemoryCompileTarget.PROMPT_FULL:
        text = adapters.compile_atom_full(atom, options)
    elif target == MemoryCompileTarget.PROMPT_INDEX:
        text = adapters.compile_atom_index(atom, options)
    elif target == MemoryCompileTarget.DENSE_EMBEDDING:
        text = adapters.compile_atom_dense(atom, options)
    elif target == MemoryCompileTarget.SPARSE_EMBEDDING:
        text = adapters.compile_atom_sparse(atom, options)
    elif target == MemoryCompileTarget.AGENT_PROFILE_MENU:
        text = adapters.compile_atom_agent_profile(atom, options)
    elif target == MemoryCompileTarget.MTP_READ:
        text = f"[{effective_alias}]:\n{atom.payload.content}"
    elif target == MemoryCompileTarget.SHARED_CONTEXT:
        text = f"[{effective_alias}]:\n{atom.payload.content}"
    elif target == MemoryCompileTarget.RUNNABLE_TOOL:
        raise ValueError("RUNNABLE_TOOL target is reserved for Phase 3.")
    else:
        raise ValueError(f"Unsupported target '{target}' for MemoryAtom source.")

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind="atom",
        alias=alias,
        memory_id=str(atom.id),
    )


def compile_pending_atom(
    pending: "PendingAtom",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """编译单个 PendingAtom。"""
    from hivememory.alice.runtime.pending_renderer import PendingAtomRenderer

    if target == MemoryCompileTarget.MTP_READ:
        text = PendingAtomRenderer.render_read(pending)
    elif target == MemoryCompileTarget.SHARED_CONTEXT:
        text = PendingAtomRenderer.render_read(pending)
    elif target == MemoryCompileTarget.MTP_ACK:
        text = PendingAtomRenderer.render_ack(pending)
    else:
        raise ValueError(f"Unsupported target '{target}' for PendingAtom source.")

    return CompiledMemoryArtifact(
        target=target,
        text=text,
        source_kind="pending",
        alias=pending.pending_alias,
        status=pending.status.value if hasattr(pending.status, "value") else str(pending.status),
    )


def compile_resolve_result(
    resolve_result: "ResolveResult",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """编译 ResolveResult，按 kind 分发。"""
    from hivememory.alice.runtime.pending_renderer import PendingAtomRenderer

    kind = resolve_result.kind

    if not options.requested_alias and resolve_result.requested_alias:
        options = options.model_copy(update={"requested_alias": resolve_result.requested_alias})
    if not options.canonical_alias and resolve_result.canonical_alias:
        options = options.model_copy(update={"canonical_alias": resolve_result.canonical_alias})

    # pending → 委托给 compile_pending_atom
    if kind == "pending" and resolve_result.pending is not None:
        return compile_pending_atom(resolve_result.pending, target, options)

    # redirect
    if kind == "redirect" and resolve_result.atom is not None:
        canonical = options.canonical_alias or resolve_result.atom.get_alias()

        if target == MemoryCompileTarget.MTP_READ:
            text = PendingAtomRenderer.render_redirect_read(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                atom=resolve_result.atom,
                settlement=resolve_result.settlement,
            )
        elif target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
            text = PendingAtomRenderer.render_redirect_run_notice(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                settlement=resolve_result.settlement,
            )
        elif target == MemoryCompileTarget.SHARED_CONTEXT:
            text = f"[{canonical}]:\n{resolve_result.atom.payload.content}"
        else:
            return compile_memory_atom(resolve_result.atom, target, options)

        return CompiledMemoryArtifact(
            target=target,
            text=text,
            source_kind="resolve_result",
            alias=canonical,
            status=kind,
        )

    # discarded / failed
    if kind in {"discarded", "failed"}:
        text = PendingAtomRenderer.render_settled_without_atom(
            requested_alias=options.requested_alias or "",
            settlement=resolve_result.settlement,
        )
        return CompiledMemoryArtifact(
            target=target,
            text=text,
            source_kind="resolve_result",
            alias=resolve_result.requested_alias,
            status=kind,
        )

    # atom (正常解析成功)
    if kind == "atom" and resolve_result.atom is not None:
        return compile_memory_atom(resolve_result.atom, target, options)

    # not_found 或其他异常状态
    raise ValueError(
        f"Cannot compile ResolveResult with kind='{kind}' for target '{target}'."
    )
