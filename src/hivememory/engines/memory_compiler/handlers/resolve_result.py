"""ResolveResult compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.builders import (
    build_memory_atom_ir,
    build_resolve_result_ir,
)
from hivememory.engines.memory_compiler.handlers.memory_atom import (
    _render_full_from_ir,
    _render_index_from_ir,
)
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n.memory_compiler import get_resolve_result_text

if TYPE_CHECKING:
    from hivememory.agent_runtime.resolver import ResolveResult
    from hivememory.core.models.pending import PendingAtomSettlement


def compile_resolve_result(
    resolve_result: "ResolveResult",
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    kind = resolve_result.kind

    if kind == "not_found":
        raise ValueError(
            f"Cannot compile ResolveResult with kind='{kind}' for target '{target}'."
        )

    # fill options aliases from resolve_result if not set
    if not options.requested_alias and resolve_result.requested_alias:
        options = options.model_copy(update={"requested_alias": resolve_result.requested_alias})
    if not options.canonical_alias and resolve_result.canonical_alias:
        options = options.model_copy(update={"canonical_alias": resolve_result.canonical_alias})

    unit = build_resolve_result_ir(resolve_result)

    # --- redirect ---
    if kind == "redirect":
        canonical = unit.identity.alias or ""
        if target == MemoryCompileTarget.MTP_READ:
            text = _t("resolve_redirect_read", options.language).format(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                status=unit.status.settlement_state or "redirected",
                content=unit.content.content or "",
            )
        elif target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
            text = _t("resolve_redirect_run_notice", options.language).format(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                status=unit.status.settlement_state or "redirected",
            )
        elif target == MemoryCompileTarget.SHARED_CONTEXT:
            # render the underlying atom at full fidelity via memory atom IR
            text = _render_full_from_ir(unit, options.max_content_length, options.stale_days, options.language)
        elif target == MemoryCompileTarget.PROMPT_FULL:
            text = _render_full_from_ir(unit, options.max_content_length, options.stale_days, options.language)
        elif target == MemoryCompileTarget.PROMPT_INDEX:
            text = _render_index_from_ir(unit, options.max_summary_length, options.stale_days, options.language)
        else:
            # other targets (DENSE_EMBEDDING etc.) — delegate to atom handler
            from hivememory.engines.memory_compiler.handlers.memory_atom import compile_memory_atom
            return compile_memory_atom(resolve_result.atom, target, options)

        return CompiledMemoryArtifact(
            target=target, text=text, source_kind="resolve_result",
            alias=canonical, status=kind,
        )

    # --- pending: delegate to pending handler with the resolved pending atom ---
    if kind == "pending" and resolve_result.pending is not None:
        from hivememory.engines.memory_compiler.handlers.pending_atom import compile_pending_atom
        return compile_pending_atom(resolve_result.pending, target, options)

    # --- atom: delegate to memory_atom handler ---
    if kind == "atom" and resolve_result.atom is not None:
        from hivememory.engines.memory_compiler.handlers.memory_atom import compile_memory_atom
        return compile_memory_atom(resolve_result.atom, target, options)

    # --- terminal: discarded / failed / expired ---
    alias = unit.identity.alias or ""
    status = unit.status

    if kind == "discarded":
        text = _t("resolve_discarded", options.language).format(
            requested_alias=alias,
            message_line=f"message: {status.message}\n" if status.message else "",
            reason_line=f"reason: {status.reason}\n" if status.reason else "",
        ).rstrip()

    elif kind == "failed":
        text = _t("resolve_failed", options.language).format(
            requested_alias=alias,
            error_line=f"error: {status.error}\n" if status.error else "",
            message_line=f"message: {status.message}\n" if status.message else "",
            reason_line=f"reason: {status.reason}\n" if status.reason else "",
        ).rstrip()

    else:  # expired
        text = _t("resolve_expired", options.language).format(requested_alias=alias)

    return CompiledMemoryArtifact(
        target=target, text=text, source_kind="resolve_result",
        alias=resolve_result.requested_alias, status=kind,
    )


def _t(key: str, language: str | None = None) -> str:
    return get_resolve_result_text(key, language)
