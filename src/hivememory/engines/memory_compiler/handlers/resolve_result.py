"""ResolveResult compilation handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.engines.memory_compiler.handlers.memory_atom import compile_memory_atom
from hivememory.engines.memory_compiler.handlers.pending_atom import compile_pending_atom
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
    """Compile a ResolveResult by dispatching on its resolution kind."""
    kind = resolve_result.kind

    if not options.requested_alias and resolve_result.requested_alias:
        options = options.model_copy(
            update={"requested_alias": resolve_result.requested_alias}
        )
    if not options.canonical_alias and resolve_result.canonical_alias:
        options = options.model_copy(
            update={"canonical_alias": resolve_result.canonical_alias}
        )

    if kind == "pending" and resolve_result.pending is not None:
        return compile_pending_atom(resolve_result.pending, target, options)

    if kind == "redirect" and resolve_result.atom is not None:
        canonical = options.canonical_alias or resolve_result.atom.get_alias()

        if target == MemoryCompileTarget.MTP_READ:
            text = _render_redirect_read(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                content=resolve_result.atom.payload.content,
                settlement=resolve_result.settlement,
                language=options.language,
            )
        elif target == MemoryCompileTarget.MTP_REDIRECT_NOTICE:
            text = _render_redirect_run_notice(
                requested_alias=options.requested_alias or "",
                canonical_alias=canonical,
                settlement=resolve_result.settlement,
                language=options.language,
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

    if kind == "discarded":
        text = _render_discarded(
            requested_alias=options.requested_alias or "",
            settlement=resolve_result.settlement,
            language=options.language,
        )
        return CompiledMemoryArtifact(
            target=target,
            text=text,
            source_kind="resolve_result",
            alias=resolve_result.requested_alias,
            status=kind,
        )

    if kind == "failed":
        text = _render_failed(
            requested_alias=options.requested_alias or "",
            settlement=resolve_result.settlement,
            language=options.language,
        )
        return CompiledMemoryArtifact(
            target=target,
            text=text,
            source_kind="resolve_result",
            alias=resolve_result.requested_alias,
            status=kind,
        )

    if kind == "expired":
        text = _render_expired(
            requested_alias=options.requested_alias or "",
            language=options.language,
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


def _render_redirect_read(
    *,
    requested_alias: str,
    canonical_alias: str,
    content: str,
    settlement: "PendingAtomSettlement | None" = None,
    language: str | None = None,
) -> str:
    status = settlement.resolution.value if settlement else "redirected"
    return get_resolve_result_text("resolve_redirect_read", language).format(
        requested_alias=requested_alias,
        canonical_alias=canonical_alias,
        status=status,
        content=content,
    )


def _render_redirect_run_notice(
    *,
    requested_alias: str,
    canonical_alias: str,
    settlement: "PendingAtomSettlement | None" = None,
    language: str | None = None,
) -> str:
    status = settlement.resolution.value if settlement else "redirected"
    return get_resolve_result_text("resolve_redirect_run_notice", language).format(
        requested_alias=requested_alias,
        canonical_alias=canonical_alias,
        status=status,
    )


def _render_discarded(
    *,
    requested_alias: str,
    settlement: "PendingAtomSettlement | None",
    language: str | None = None,
) -> str:
    message = settlement.message if settlement and settlement.message else ""
    reason = settlement.reason if settlement and settlement.reason else ""
    return get_resolve_result_text("resolve_discarded", language).format(
        requested_alias=requested_alias,
        message_line=f"message: {message}\n" if message else "",
        reason_line=f"reason: {reason}\n" if reason else "",
    ).rstrip()


def _render_failed(
    *,
    requested_alias: str,
    settlement: "PendingAtomSettlement | None",
    language: str | None = None,
) -> str:
    error = settlement.error if settlement and settlement.error else ""
    message = settlement.message if settlement and settlement.message else ""
    reason = settlement.reason if settlement and settlement.reason else ""
    return get_resolve_result_text("resolve_failed", language).format(
        requested_alias=requested_alias,
        error_line=f"error: {error}\n" if error else "",
        message_line=f"message: {message}\n" if message else "",
        reason_line=f"reason: {reason}\n" if reason else "",
    ).rstrip()


def _render_expired(
    *,
    requested_alias: str,
    language: str | None = None,
) -> str:
    return get_resolve_result_text("resolve_expired", language).format(
        requested_alias=requested_alias,
    )
