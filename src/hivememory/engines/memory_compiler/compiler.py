"""MemoryCompiler 统一入口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

from hivememory.config.memory_compiler import CompactContextStrategyConfig
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
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.i18n.resolver import get_default_language

if TYPE_CHECKING:
    from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement
    from hivememory.core.models.reference import ReferenceResolution

    # 公开编译源；运行时由 ``_build_unit_ir`` 按同一集合分派。
    CompilableMemory = MemoryAtom | PendingAtom | ReferenceResolution | PendingAtomSettlement


class MemoryCompiler:
    """将可编译记忆数据转为 Agent 可读文本。"""

    # 单个源经单元 target 恰好产出一个单元；列表源长度为 1 时同样返回单个
    # 产物，因此列表源只能声明为联合；信封 target 总是产出一份文本。
    @overload
    def compile(
        self,
        source: CompilableMemory,
        target: MemoryCompileTarget,
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemory: ...

    @overload
    def compile[SourceT: CompilableMemory](
        self,
        source: list[SourceT],
        target: MemoryCompileTarget,
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemory | list[CompiledMemory]: ...

    @overload
    def compile[SourceT: CompilableMemory](
        self,
        source: CompilableMemory | list[SourceT],
        target: MemoryEnvelopeTarget,
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemory: ...

    def compile(
        self,
        source: CompilableMemory | Sequence[CompilableMemory],
        target: MemoryCompileTarget | MemoryEnvelopeTarget,
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
            opts = opts.model_copy(
                update={"retrieval_strategy_config": CompactContextStrategyConfig()}
            )
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

    def _build_unit_ir(
        self, source, options: MemoryCompileOptions
    ) -> tuple[MemoryUnitIR, MemoryCompileOptions]:
        """
        从可编译单元 source 构建 MemoryUnitIR。
        """
        from hivememory.core.models.pending import (
            PendingAtom,
            PendingAtomResolution,
            PendingAtomSettlement,
        )
        from hivememory.core.models.reference import ReferenceResolution

        if isinstance(source, ReferenceResolution):
            if source.kind == "not_found":
                raise ValueError(
                    "Cannot compile ReferenceResolution with kind='not_found': it has no "
                    "content; callers report the alias as unresolved."
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
            kind: Literal["discarded", "redirect"] = (
                "discarded" if source.resolution == PendingAtomResolution.DISCARDED else "redirect"
            )
            resolve = ReferenceResolution(
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
        if isinstance(source, ReferenceResolution):
            return build_resolve_result_ir(source), options

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            "Expected MemoryAtom, PendingAtom, ReferenceResolution, or PendingAtomSettlement."
        )

    def _build_unit_irs(
        self, source, options: MemoryCompileOptions
    ) -> tuple[list[MemoryUnitIR], MemoryCompileOptions]:
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


def readable_source(
    result: ReferenceResolution,
) -> PendingAtom | ReferenceResolution | MemoryAtom | None:
    """READ 交付给编译器的内容；无可交付内容时返回 ``None``，按未解析提示。

    pending 交付意图句柄；带可读原子的 redirect 与 discarded/failed/expired
    终态交付解析结果本身；正式原子交付原子；其余（not_found、不可读的
    redirect、缺失载荷的 pending）都按未解析处理。MTP READ 与 MCP 的读取
    工具共用这一份规则。
    """
    if result.kind == "pending":
        return result.pending
    if result.kind == "redirect":
        return result if result.atom is not None else None
    if result.kind == "atom":
        return result.atom
    if result.kind in {"discarded", "failed", "expired"}:
        return result
    return None
