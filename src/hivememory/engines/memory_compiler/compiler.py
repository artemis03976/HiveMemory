"""MemoryCompiler 统一入口。"""

from __future__ import annotations

from typing import Tuple, Union

from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.engines.memory_compiler.builders import (
    build_memory_atom_ir,
    build_pending_atom_ir,
    build_resolve_result_ir,
)
from hivememory.engines.memory_compiler.envelopes import compile_envelope_from_ir
from hivememory.engines.memory_compiler.handlers import compile_unit_from_ir
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR, MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemory,
    FullStrategyConfig,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.i18n.resolver import get_default_language


class MemoryCompiler:
    """将可编译记忆数据转为 Agent 可读文本。"""

    def compile(
        self,
        source,
        target: Union[MemoryCompileTarget, MemoryEnvelopeTarget],
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemory | list[CompiledMemory]:
        """
        统一端到端编译入口。

        接收可被编译的记忆类数据结构，按照指定 target 直接产出对应的 Agent 可理解形式。
        """
        # 拒绝非记忆类数据结构作为编译源
        self._reject_public_unsupported_source(source)

        # 为编译选项添加兜底
        opts = self._prepare_options(options, target)

        # 统一将所有记忆数据转换为 MemoryUnitIR 
        unit_irs, effective_options = self._build_unit_irs(source, opts)

        if isinstance(target, MemoryCompileTarget):
            return self._handle_unit_target(unit_irs, target, effective_options)

        # 处理额外的列表包裹结构
        if isinstance(target, MemoryEnvelopeTarget):
            return self._handle_envelope_target(unit_irs, target, effective_options)

        raise TypeError(f"Unsupported compile target: {target!r}")

    def _reject_public_unsupported_source(self, source) -> None:
        """
        拒绝公开的内部中间结果作为编译源。
        """
        unsupported_types = (
            MemoryUnitIR,
            MemorySectionIR,
            MemoryBundleIR,
            CompiledMemory,
            MemoryEnvelopeSection,
        )
        if isinstance(source, unsupported_types):
            raise TypeError(
                f"Unsupported source type: {type(source).__name__}. "
                "IR objects and compiled memory outputs are internal MemoryCompiler intermediates."
            )
        if isinstance(source, list):
            for item in source:
                self._reject_public_unsupported_source(item)

    def _prepare_options(
        self,
        options: MemoryCompileOptions | None,
        target: MemoryCompileTarget | MemoryEnvelopeTarget,
    ) -> MemoryCompileOptions:
        """
        为编译选项增加兜底逻辑。
        """
        opts = options or MemoryCompileOptions()

        if opts.language is None:
            opts = opts.model_copy(update={"language": get_default_language().value})
        
        if (
            target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT
            and opts.retrieval_strategy_config is None
        ):
            opts = opts.model_copy(update={"retrieval_strategy_config": FullStrategyConfig()})
        return opts

    def _handle_unit_target(
        self,
        unit_irs: list[MemoryUnitIR],
        target: MemoryCompileTarget,
        options: MemoryCompileOptions,
    ) -> CompiledMemory | list[CompiledMemory]:
        """
        单记忆编译处理管线。

        负责将 MemoryUnitIR 编译为 Agent 可读文本。
        """
        if len(unit_irs) == 1:
            return compile_unit_from_ir(unit_irs[0], target, options)
        return [compile_unit_from_ir(unit_ir, target, options) for unit_ir in unit_irs]

    def _handle_envelope_target(
        self,
        unit_irs: list[MemoryUnitIR],
        target: MemoryEnvelopeTarget,
        options: MemoryCompileOptions,
    ) -> CompiledMemory:
        """
        多记忆编译处理管线。

        将 MemoryUnitIR 组装为 section/bundle IR，再编译为 Agent 可读文本。
        """
        section_irs = self._build_section_irs_from_units(unit_irs, target)
        bundle_ir = self._build_bundle_ir(section_irs, target, options)
        return compile_envelope_from_ir(bundle_ir, options=options)

    def _build_unit_ir(self, source, options: MemoryCompileOptions) -> Tuple[MemoryUnitIR, MemoryCompileOptions]:
        """
        从可编译单元 source 构建 MemoryUnitIR。
        """
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models.pending import PendingAtom, PendingAtomResolution, PendingAtomSettlement

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
            "Expected MemoryAtom, PendingAtom, ResolveResult, or PendingAtomSettlement."
        )

    def _build_unit_irs(self, source, options: MemoryCompileOptions) -> tuple[list[MemoryUnitIR], MemoryCompileOptions]:
        if isinstance(source, list):
            units: list[MemoryUnitIR] = []
            effective_options = options
            for item in source:
                unit, effective_options = self._build_unit_ir(item, effective_options)
                units.append(unit)
            return units, effective_options

        unit, effective_options = self._build_unit_ir(source, options)
        return [unit], effective_options

    def _build_section_irs_from_units(
        self,
        units: list[MemoryUnitIR],
        target: MemoryEnvelopeTarget,
    ) -> list[MemorySectionIR]:
        if target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT:
            return self._build_retrieval_context_sections(units)
        return self._build_default_sections(units)

    def _build_retrieval_context_sections(
        self,
        units: list[MemoryUnitIR],
    ) -> list[MemorySectionIR]:
        memories: list[MemoryUnitIR] = []
        agent_profiles: list[MemoryUnitIR] = []
        for unit in units:
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

    def _build_default_sections(
        self,
        units: list[MemoryUnitIR],
    ) -> list[MemorySectionIR]:
        return [MemorySectionIR(kind="default", units=units)] if units else []

    def _build_bundle_ir(
        self,
        sections: list[MemorySectionIR],
        target: MemoryEnvelopeTarget,
        options: MemoryCompileOptions,
    ) -> MemoryBundleIR:
        metadata = {"format": options.format} if options.format else {}
        return MemoryBundleIR(purpose=target, sections=sections, metadata=metadata)
