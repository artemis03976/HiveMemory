"""MemoryCompiler — 记忆到文本编译的统一入口。"""

from __future__ import annotations

from typing import List, Union

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
from hivememory.engines.memory_compiler.handlers import (
    compile_memory_atom,
    compile_pending_atom,
    compile_resolve_result,
)


class MemoryCompiler:
    """
    记忆编译外观层。

    Phase 1: 委托给现有渲染器，收敛所有入口。
    Phase 2+: 内化渲染逻辑，引入 IR 和缓存。
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
        from hivememory.alice.runtime.resolver import ResolveResult
        from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement

        if isinstance(source, MemoryAtom):
            return compile_memory_atom(source, target, options)

        if isinstance(source, PendingAtom):
            return compile_pending_atom(source, target, options)

        if isinstance(source, ResolveResult):
            return compile_resolve_result(source, target, options)

        if isinstance(source, PendingAtomSettlement):
            from hivememory.core.models.pending import PendingAtomResolution

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
            return compile_resolve_result(resolve, target, options)

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            f"Expected MemoryAtom, PendingAtom, ResolveResult, or PendingAtomSettlement."
        )
