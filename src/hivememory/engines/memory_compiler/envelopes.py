"""Envelope compilation for compiled memory artifacts."""

from __future__ import annotations

from typing import Iterable

from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    CompiledMemoryEnvelope,
    MemoryCompileOptions,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)


def compile_envelope(
    target: MemoryEnvelopeTarget,
    *,
    artifacts: Iterable[CompiledMemoryArtifact] | None = None,
    sections: list[MemoryEnvelopeSection] | None = None,
    options: MemoryCompileOptions | None = None,
) -> CompiledMemoryEnvelope:
    """Compile envelope text around already-compiled memory artifacts."""
    opts = options or MemoryCompileOptions()
    envelope_sections = sections or [
        MemoryEnvelopeSection(kind="default", artifacts=list(artifacts or []))
    ]

    if target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT:
        text = _compile_retrieval_context(envelope_sections)
    elif target == MemoryEnvelopeTarget.MTP_READ_RESPONSE:
        text = _compile_mtp_read_response(envelope_sections)
    elif target == MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION:
        text = _compile_shared_context_injection(envelope_sections)
    else:
        raise ValueError(f"Unsupported envelope target '{target}'.")

    return CompiledMemoryEnvelope(
        target=target,
        text=text,
        sections=envelope_sections,
        metadata={"format": opts.format} if opts.format else {},
    )


def _compile_retrieval_context(sections: list[MemoryEnvelopeSection]) -> str:
    from hivememory.utils.memory_atom_renderer import MEMORY_FOOTER, MEMORY_HEADER

    parts = [MEMORY_HEADER]
    for section in sections:
        if section.kind == "memories":
            parts.append(_render_retrieval_memories_section(section))
        elif section.kind == "agent_profiles":
            parts.append(_render_retrieval_agent_profiles_section(section))
        elif section.artifacts:
            parts.append("".join(artifact.text for artifact in section.artifacts))
        elif section.empty_text:
            parts.append(section.empty_text)
    parts.append(MEMORY_FOOTER)
    return "".join(parts)


def _render_retrieval_memories_section(section: MemoryEnvelopeSection) -> str:
    if section.artifacts:
        return "\n### 相关记忆 (Relevant Memories)\n" + "".join(
            artifact.text for artifact in section.artifacts
        )
    if section.empty_text:
        return f"\n### 相关记忆 (Relevant Memories)\n{section.empty_text}"
    return ""


def _render_retrieval_agent_profiles_section(section: MemoryEnvelopeSection) -> str:
    if section.artifacts:
        lines = ["\n### 可用子代理 (Available Sub-Agents)"]
        lines.extend(artifact.text for artifact in section.artifacts)
        return "\n".join(lines)
    if section.empty_text:
        return f"\n### 可用子代理 (Available Sub-Agents)\n{section.empty_text}"
    return ""


def _compile_mtp_read_response(sections: list[MemoryEnvelopeSection]) -> str:
    body = _join_section_artifacts(sections)
    return "[MTP READ Result]\n" + body if body else "[MTP READ Result]"


def _compile_shared_context_injection(sections: list[MemoryEnvelopeSection]) -> str:
    body = _join_section_artifacts(sections)
    if not body:
        return "[Shared Context]\nNo shared memory artifacts."
    return (
        "[Shared Context]\n"
        "Runtime memory artifacts are available below and may be inspected with READ.\n"
        + body
    )


def _join_section_artifacts(sections: list[MemoryEnvelopeSection]) -> str:
    blocks: list[str] = []
    for section in sections:
        if section.artifacts:
            blocks.extend(artifact.text for artifact in section.artifacts)
        elif section.empty_text:
            blocks.append(section.empty_text)
    return "\n".join(blocks)
