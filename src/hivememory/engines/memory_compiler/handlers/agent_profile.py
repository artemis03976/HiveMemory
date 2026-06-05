"""Agent profile menu target handler."""

from __future__ import annotations

from hivememory.engines.memory_compiler.handlers.common import build_artifact
from hivememory.engines.memory_compiler.ir import MemoryUnitIR
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.i18n import get_memory_atom_text


def compile_agent_profile_menu(
    unit: MemoryUnitIR,
    target: MemoryCompileTarget,
    options: MemoryCompileOptions,
) -> CompiledMemoryArtifact:
    if unit.identity.source_kind != "atom":
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(
        unit,
        target,
        _render_agent_profile_from_ir(unit, language=options.language),
        options,
    )


def _render_agent_profile_from_ir(unit: MemoryUnitIR, language: str | None = None) -> str:
    alias = unit.identity.alias or ""
    title = unit.content.title or _text("memory_agent_profile_untitled", language)
    summary = unit.content.summary or ""
    return _text("memory_agent_profile_item_template", language).format(
        alias=alias,
        title=title,
        summary=summary,
    )


def _text(key: str, language: str | None = None) -> str:
    return get_memory_atom_text(key, language)
