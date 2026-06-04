"""MemoryCompiler — 记忆到文本编译的统一入口。"""

from __future__ import annotations

from typing import List, Tuple, Union

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler.envelopes import compile_envelope
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryEnvelope,
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.engines.memory_compiler.builders import (
    build_memory_atom_ir,
    build_pending_atom_ir,
    build_resolve_result_ir,
)
from hivememory.engines.memory_compiler.ir import MemoryUnitIR


class MemoryCompiler:
    """
    记忆编译外观层。

    Phase 2B: source -> IR -> target 的完整收束。
    """

    def __init__(self, default_language: str = "zh") -> None:
        self.default_language = default_language

    def compile(
        self,
        source: Union[
            MemoryAtom,
            "PendingAtom",
            "ResolveResult",
            "PendingAtomSettlement",
            List[MemoryAtom],
            List["ResolveResult"],
        ],
        target: MemoryCompileTarget,
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemoryArtifact | List[CompiledMemoryArtifact]:
        opts = self._resolve_options(options)

        if isinstance(source, list):
            return [self._compile_single(item, target, opts) for item in source]

        return self._compile_single(source, target, opts)

    def wrap(
        self,
        artifacts: CompiledMemoryArtifact | List[CompiledMemoryArtifact] | None = None,
        envelope_target: MemoryEnvelopeTarget = MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        options: MemoryCompileOptions | None = None,
        sections: List[MemoryEnvelopeSection] | None = None,
    ) -> CompiledMemoryEnvelope:
        artifact_list: list[CompiledMemoryArtifact]
        if artifacts is None:
            artifact_list = []
        elif isinstance(artifacts, list):
            artifact_list = artifacts
        else:
            artifact_list = [artifacts]

        return compile_envelope(
            envelope_target,
            artifacts=artifact_list,
            sections=sections,
            options=self._resolve_options(options),
        )

    def _resolve_options(self, options: MemoryCompileOptions | None) -> MemoryCompileOptions:
        opts = options or MemoryCompileOptions()
        if opts.language is None:
            opts = opts.model_copy(update={"language": self.default_language})
        return opts

    def _compile_single(
        self,
        source,
        target: MemoryCompileTarget,
        options: MemoryCompileOptions,
    ) -> CompiledMemoryArtifact:
        from hivememory.engines.memory_compiler.handlers import compile_from_ir

        unit, effective_options = self._build_unit_ir(source, options)
        return compile_from_ir(unit, target, effective_options)

    def _build_unit_ir(self, source, options: MemoryCompileOptions) -> Tuple[MemoryUnitIR, MemoryCompileOptions]:
        """Phase 2B: build MemoryUnitIR from any supported source."""
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement, PendingAtomResolution

        # ResolveResult 透明展开
        if isinstance(source, ResolveResult):
            if source.kind == "not_found":
                raise ValueError(
                    "Cannot compile ResolveResult with kind='not_found'. "
                    "Handle this in MTP runtime instead."
                )
            if source.kind == "pending" and source.pending:
                if not options.requested_alias:
                    options = options.model_copy(update={"requested_alias": source.requested_alias})
                source = source.pending
            elif source.kind == "atom" and source.atom:
                if not options.requested_alias:
                    options = options.model_copy(update={"requested_alias": source.requested_alias})
                source = source.atom
            # redirect / terminal 继续走 build_resolve_result_ir

        # PendingAtomSettlement wrapper
        if isinstance(source, PendingAtomSettlement):
            kind = (
                "discarded"
                if source.resolution == PendingAtomResolution.DISCARDED
                else "redirect"
            )
            resolve = ResolveResult(
                kind=kind,
                requested_alias=source.pending_alias,
                canonical_alias=source.canonical_alias,
                settlement=source,
            )
            return build_resolve_result_ir(resolve), options

        # 三个 builder
        if isinstance(source, MemoryAtom):
            return build_memory_atom_ir(source), options
        if isinstance(source, PendingAtom):
            return build_pending_atom_ir(source), options
        if isinstance(source, ResolveResult):
            return build_resolve_result_ir(source), options

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            f"Expected MemoryAtom, PendingAtom, ResolveResult, or PendingAtomSettlement."
        )
