"""Envelope compilation for compiled memory artifacts."""

from __future__ import annotations

from typing import List

from hivememory.i18n import (
    get_memory_envelope_text,
    get_memory_section_title,
)
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR, MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemory,
    CompiledMemoryArtifact,
    CompiledMemoryEnvelope,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.system.config.memory_compiler import (
    CascadeContextStrategyConfig,
    CompactContextStrategyConfig,
    FullContextStrategyConfig,
)
from hivememory.utils import estimate_tokens


def compile_envelope_from_ir(
    bundle: MemoryBundleIR,
    *,
    options: MemoryCompileOptions | None = None,
) -> CompiledMemoryEnvelope:
    """
    将 MemoryBundleIR 编译为 envelope target 的终态文本。
    """
    opts = options or MemoryCompileOptions()

    # A2: 如果 section 有 units，按策略编译为 artifacts
    resolved_sections = [_resolve_section_units(s, opts) for s in bundle.sections]

    if bundle.purpose == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT:
        text = _compile_retrieval_context(resolved_sections, opts.language)
    elif bundle.purpose == MemoryEnvelopeTarget.MTP_READ_RESPONSE:
        text = _compile_mtp_read_response(resolved_sections, opts.language)
    elif bundle.purpose == MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION:
        text = _compile_shared_context_injection(resolved_sections, opts.language)
    else:
        raise ValueError(f"Unsupported envelope target '{bundle.purpose}'.")

    metadata = dict(bundle.metadata)
    if opts.format:
        metadata["format"] = opts.format

    return CompiledMemory(
        target=bundle.purpose,
        text=text,
        sections=[_to_envelope_section(section) for section in resolved_sections],
        metadata=metadata,
    )


# ========== A2: 策略编译 ==========

def _resolve_section_units(
    section: MemorySectionIR,
    options: MemoryCompileOptions,
) -> MemorySectionIR:
    """将 section.units 按检索策略编译为 artifacts；已有 artifacts 则直接返回。"""
    if not section.units:
        return section
    if section.kind == "agent_profiles":
        artifacts = _compile_units_for_target(
            section.units,
            MemoryCompileTarget.AGENT_PROFILE_MENU,
            options,
        )
    elif section.kind == "default":
        artifacts = _compile_units_for_target(
            section.units,
            MemoryCompileTarget.SHARED_CONTEXT,
            options,
        )
    else:
        artifacts = _compile_units_with_strategy(section.units, options)
    return section.model_copy(update={"artifacts": artifacts})


def _compile_units_for_target(
    units: List[MemoryUnitIR],
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> List[CompiledMemoryArtifact]:
    from hivememory.engines.memory_compiler.handlers.targets import compile_unit_from_ir

    return [compile_unit_from_ir(unit, target, options) for unit in units]


def _compile_units_with_strategy(
    units: List[MemoryUnitIR],
    options: MemoryCompileOptions,
) -> List[CompiledMemoryArtifact]:
    from hivememory.engines.memory_compiler.handlers.targets import compile_unit_from_ir

    cfg = options.retrieval_strategy_config

    if isinstance(cfg, FullContextStrategyConfig):
        return _apply_full_strategy(units, cfg, options, compile_unit_from_ir)
    if isinstance(cfg, CascadeContextStrategyConfig):
        return _apply_cascade_strategy(units, cfg, options, compile_unit_from_ir)
    if isinstance(cfg, CompactContextStrategyConfig):
        return _apply_compact_strategy(units, cfg, options, compile_unit_from_ir)

    # 无策略配置：全量 PROMPT_FULL
    return [compile_unit_from_ir(u, MemoryCompileTarget.PROMPT_FULL, options) for u in units]


def _apply_full_strategy(units, cfg: FullContextStrategyConfig, opts, compile_unit_from_ir):
    artifacts: List[CompiledMemoryArtifact] = []
    total = 0
    unit_opts = opts.model_copy(update={
        "max_content_length": cfg.max_content_length,
        "stale_days": cfg.stale_days,
    })
    for unit in units:
        artifact = compile_unit_from_ir(unit, MemoryCompileTarget.PROMPT_FULL, unit_opts)
        if total + len(artifact.text) > cfg.max_tokens:
            break
        artifacts.append(artifact)
        total += len(artifact.text)
    return artifacts


def _apply_cascade_strategy(units, cfg: CascadeContextStrategyConfig, opts, compile_unit_from_ir):
    artifacts: List[CompiledMemoryArtifact] = []
    remaining = cfg.max_memory_tokens
    for i, unit in enumerate(units):
        if i < cfg.full_payload_count:
            a = compile_unit_from_ir(
                unit,
                MemoryCompileTarget.PROMPT_FULL,
                opts.model_copy(update={"max_content_length": cfg.max_content_length}),
            )
            tokens = estimate_tokens(a.text)
            if tokens <= remaining:
                artifacts.append(a)
                remaining -= tokens
                continue
        a = compile_unit_from_ir(
            unit,
            MemoryCompileTarget.PROMPT_INDEX,
            opts.model_copy(update={"max_summary_length": cfg.index_max_summary_length}),
        )
        tokens = estimate_tokens(a.text)
        if tokens <= remaining:
            artifacts.append(a)
            remaining -= tokens
        else:
            break
    return artifacts


def _apply_compact_strategy(units, cfg: CompactContextStrategyConfig, opts, compile_unit_from_ir):
    artifacts: List[CompiledMemoryArtifact] = []
    remaining = cfg.max_memory_tokens
    index_opts = opts.model_copy(update={"max_summary_length": cfg.index_max_summary_length})
    for unit in units:
        a = compile_unit_from_ir(unit, MemoryCompileTarget.PROMPT_INDEX, index_opts)
        tokens = estimate_tokens(a.text)
        if tokens <= remaining:
            artifacts.append(a)
            remaining -= tokens
        else:
            break
    return artifacts


# ========== Envelope rendering ==========

def _compile_retrieval_context(
    sections: list[MemorySectionIR],
    language: str | None,
) -> str:
    parts = [get_memory_envelope_text("retrieval_header", language)]
    for section in sections:
        if section.kind == "memories":
            parts.append(_render_retrieval_memories_section(section, language))
        elif section.kind == "agent_profiles":
            parts.append(_render_retrieval_agent_profiles_section(section, language))
        elif section.artifacts:
            parts.append("".join(artifact.text for artifact in section.artifacts))
        elif section.empty_text:
            parts.append(section.empty_text)
    parts.append(get_memory_envelope_text("retrieval_footer", language))
    return "".join(parts)


def _render_retrieval_memories_section(
    section: MemorySectionIR,
    language: str | None,
) -> str:
    title = get_memory_section_title("memories", language)
    if section.artifacts:
        return f"\n### {title}\n" + "".join(
            artifact.text for artifact in section.artifacts
        )
    if section.empty_text:
        return f"\n### {title}\n{section.empty_text}"
    return ""


def _render_retrieval_agent_profiles_section(
    section: MemorySectionIR,
    language: str | None,
) -> str:
    title = get_memory_section_title("agent_profiles", language)
    if section.artifacts:
        return f"\n### {title}\n" + "".join(
            artifact.text for artifact in section.artifacts
        )
    if section.empty_text:
        return f"\n### {title}\n{section.empty_text}"
    return ""


def _compile_mtp_read_response(
    sections: list[MemorySectionIR],
    language: str | None,
) -> str:
    body = _join_section_artifacts(sections)
    title = get_memory_envelope_text("mtp_read_result_title", language)
    return f"{title}\n{body}" if body else title


def _compile_shared_context_injection(
    sections: list[MemorySectionIR],
    language: str | None,
) -> str:
    body = _join_section_artifacts(sections)
    title = get_memory_envelope_text("shared_context_title", language)
    if not body:
        empty = get_memory_envelope_text("shared_context_empty", language)
        return f"{title}\n{empty}"
    intro = get_memory_envelope_text("shared_context_intro", language)
    return f"{title}\n\n{intro}\n\n{body}"


def _join_section_artifacts(sections: list[MemorySectionIR]) -> str:
    blocks: list[str] = []
    for section in sections:
        if section.artifacts:
            blocks.extend(artifact.text for artifact in section.artifacts)
        elif section.empty_text:
            blocks.append(section.empty_text)
    return "\n".join(blocks)


def _to_envelope_section(section: MemorySectionIR) -> MemoryEnvelopeSection:
    return MemoryEnvelopeSection(
        kind=section.kind,
        artifacts=list(section.artifacts),
        empty_text=section.empty_text,
    )
