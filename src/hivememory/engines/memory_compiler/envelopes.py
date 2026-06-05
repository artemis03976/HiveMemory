"""Envelope compilation for compiled memory artifacts."""

from __future__ import annotations

from hivememory.i18n import (
    get_memory_envelope_text,
    get_memory_section_title,
)
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryEnvelope,
    MemoryCompileOptions,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)


def compile_envelope(
    bundle: MemoryBundleIR,
    *,
    options: MemoryCompileOptions | None = None,
) -> CompiledMemoryEnvelope:
    """Compile envelope text around a memory bundle."""
    opts = options or MemoryCompileOptions()

    if bundle.purpose == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT:
        text = _compile_retrieval_context(bundle.sections, opts.language)
    elif bundle.purpose == MemoryEnvelopeTarget.MTP_READ_RESPONSE:
        text = _compile_mtp_read_response(bundle.sections, opts.language)
    elif bundle.purpose == MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION:
        text = _compile_shared_context_injection(bundle.sections, opts.language)
    else:
        raise ValueError(f"Unsupported envelope target '{bundle.purpose}'.")

    metadata = dict(bundle.metadata)
    if opts.format:
        metadata["format"] = opts.format

    return CompiledMemoryEnvelope(
        target=bundle.purpose,
        text=text,
        sections=[_to_envelope_section(section) for section in bundle.sections],
        metadata=metadata,
    )


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
