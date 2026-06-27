"""Unified MemoryCompiler entrypoint."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.engines.memory_compiler.builders import (
    build_memory_atom_ir,
    build_pending_atom_ir,
    build_resolve_result_ir,
)
from hivememory.engines.memory_compiler.envelopes import compile_envelope
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR, MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemory,
    FullStrategyConfig,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeTarget,
)
from hivememory.i18n.resolver import get_default_language


class MemoryCompiler:
    """Memory to agent-readable text compiler."""

    def compile(
        self,
        source,
        target: Union[MemoryCompileTarget, MemoryEnvelopeTarget],
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemory | list[CompiledMemory]:
        opts = self._resolve_options(options)

        if isinstance(target, MemoryEnvelopeTarget):
            bundle, opts = self._build_bundle_ir_from_source(source, target, opts)
            return compile_envelope(bundle, options=opts)

        if isinstance(source, list):
            return [self._compile_single(item, target, opts) for item in source]

        return self._compile_single(source, target, opts)

    def _resolve_options(self, options: MemoryCompileOptions | None) -> MemoryCompileOptions:
        opts = options or MemoryCompileOptions()
        if opts.language is None:
            opts = opts.model_copy(update={"language": get_default_language().value})
        return opts

    def _compile_single(
        self,
        source,
        target: MemoryCompileTarget,
        options: MemoryCompileOptions,
    ) -> CompiledMemory:
        from hivememory.engines.memory_compiler.handlers import compile_from_ir

        unit, effective_options = self._build_unit_ir(source, options)
        return compile_from_ir(unit, target, effective_options)

    def _build_unit_ir(self, source, options: MemoryCompileOptions) -> Tuple[MemoryUnitIR, MemoryCompileOptions]:
        """Build MemoryUnitIR from any supported unit source."""
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models.pending import PendingAtom, PendingAtomResolution, PendingAtomSettlement

        if isinstance(source, MemoryUnitIR):
            return source, options

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

        if isinstance(source, MemoryAtom):
            return build_memory_atom_ir(source), options
        if isinstance(source, PendingAtom):
            return build_pending_atom_ir(source), options
        if isinstance(source, ResolveResult):
            return build_resolve_result_ir(source), options

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            "Expected MemoryAtom, PendingAtom, ResolveResult, PendingAtomSettlement, or MemoryUnitIR."
        )

    def _build_bundle_ir_from_source(
        self,
        source,
        target: MemoryEnvelopeTarget,
        options: MemoryCompileOptions,
    ) -> tuple[MemoryBundleIR, MemoryCompileOptions]:
        metadata = {"format": options.format} if options.format else {}

        if isinstance(source, MemoryBundleIR):
            return source, self._default_retrieval_strategy(target, options)

        if isinstance(source, MemorySectionIR):
            bundle = MemoryBundleIR(purpose=target, sections=[source], metadata=metadata)
            return bundle, self._default_retrieval_strategy(target, options)

        if isinstance(source, list) and all(isinstance(item, MemorySectionIR) for item in source):
            bundle = MemoryBundleIR(purpose=target, sections=list(source), metadata=metadata)
            return bundle, self._default_retrieval_strategy(target, options)

        if isinstance(source, list) and all(isinstance(item, CompiledMemory) for item in source):
            bundle = MemoryBundleIR(
                purpose=target,
                sections=[MemorySectionIR(kind="default", artifacts=list(source))],
                metadata=metadata,
            )
            return bundle, self._default_retrieval_strategy(target, options)

        if isinstance(source, CompiledMemory):
            bundle = MemoryBundleIR(
                purpose=target,
                sections=[MemorySectionIR(kind="default", artifacts=[source])],
                metadata=metadata,
            )
            return bundle, self._default_retrieval_strategy(target, options)

        units, effective_options = self._build_units(source, options)
        sections = self._build_sections_for_target(units, target)
        bundle = MemoryBundleIR(purpose=target, sections=sections, metadata=metadata)
        return bundle, self._default_retrieval_strategy(target, effective_options)

    def _build_units(self, source, options: MemoryCompileOptions) -> tuple[list[MemoryUnitIR], MemoryCompileOptions]:
        if source is None:
            return [], options

        if isinstance(source, MemoryUnitIR):
            return [source], options

        if isinstance(source, list):
            units: list[MemoryUnitIR] = []
            effective_options = options
            for item in source:
                unit, effective_options = self._build_unit_ir(item, effective_options)
                units.append(unit)
            return units, effective_options

        unit, effective_options = self._build_unit_ir(source, options)
        return [unit], effective_options

    def _build_sections_for_target(
        self,
        units: Iterable[MemoryUnitIR],
        target: MemoryEnvelopeTarget,
    ) -> list[MemorySectionIR]:
        unit_list = list(units)
        if target != MemoryEnvelopeTarget.RETRIEVAL_CONTEXT:
            return [MemorySectionIR(kind="default", units=unit_list)] if unit_list else []

        memories: list[MemoryUnitIR] = []
        agent_profiles: list[MemoryUnitIR] = []
        for unit in unit_list:
            if unit.content.memory_type == MemoryType.AGENT_PROFILE.value:
                agent_profiles.append(unit)
            else:
                memories.append(unit)

        sections: list[MemorySectionIR] = []
        if memories:
            sections.append(MemorySectionIR(kind="memories", units=memories))
        if agent_profiles:
            sections.append(MemorySectionIR(kind="agent_profiles", units=agent_profiles))
        return sections

    def _default_retrieval_strategy(
        self,
        target: MemoryEnvelopeTarget,
        options: MemoryCompileOptions,
    ) -> MemoryCompileOptions:
        if (
            target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT
            and options.retrieval_strategy_config is None
        ):
            return options.model_copy(update={"retrieval_strategy_config": FullStrategyConfig()})
        return options
