"""ResolveResult compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.handlers.memory_atom import compile_memory_atom
from hivememory.engines.memory_compiler.handlers.pending_atom import (
    _render_redirect_read,
    _render_redirect_run_notice,
    _render_settled_without_atom,
    compile_pending_atom,
)
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.resolver import ResolveResult


def compile_resolve_result(
    resolve_result: "ResolveResult",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    """Compile a ResolveResult by dispatching on its resolution kind."""
    kind = resolve_result.kind

    if not options.requested_alias and resolve_result.requested_alias:
        options = options.model_copy(update={"requested_alias": resolve_result.requested_alias})
    if not options.canonical_alias and resolve_result.canonical_alias:
        options = options.model_copy(update={"canonical_alias": resolve_result.canonical_alias})

    if kind == "pending" and resolve_result.pending is not None:
        return compile_pending_atom(resolve_result.pending, target, options)

    if kind == "redirect" and resolve_result.atom is not None:
        canonical = options.canonical_alias or resolve_result.atom.get_alias()

        if target == MemoryCompileTarget.MTP_READ:
            text = _render_redirect_read(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                atom=resolve_result.atom,
                settlement=resolve_result.settlement,
            )
        elif target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
            text = _render_redirect_run_notice(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                settlement=resolve_result.settlement,
            )
        elif target == MemoryCompileTarget.SHARED_CONTEXT:
            artifact = compile_memory_atom(resolve_result.atom, target, options)
            text = artifact.text
        else:
            return compile_memory_atom(resolve_result.atom, target, options)

        return CompiledMemoryArtifact(
            target=target,
            text=text,
            source_kind="resolve_result",
            alias=canonical,
            status=kind,
        )

    if kind in {"discarded", "failed"}:
        text = _render_settled_without_atom(
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

    if kind == "atom" and resolve_result.atom is not None:
        return compile_memory_atom(resolve_result.atom, target, options)

    raise ValueError(
        f"Cannot compile ResolveResult with kind='{kind}' for target '{target}'."
    )
