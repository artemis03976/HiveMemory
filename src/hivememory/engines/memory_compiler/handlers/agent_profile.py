"""可用 agent 列表的菜单项 target 处理器。"""

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
    if unit.identity.source_kind != "agent_profile":
        raise ValueError(f"Unsupported source '{unit.identity.source_kind}' for target '{target}'.")
    return build_artifact(
        unit,
        target,
        _render_agent_profile_from_ir(unit, language=options.language),
        options,
    )


def _render_agent_profile_from_ir(unit: MemoryUnitIR, language: str | None = None) -> str:
    """每项是 agent_id 加上一句描述：用户给它起的名字与它负责的功能拼成“title: summary”。"""
    title = unit.content.title or ""
    summary = unit.content.summary or ""
    if title and summary:
        description = f"{title}: {summary}"
    else:
        description = title or summary or _text("memory_agent_profile_untitled", language)
    return _text("memory_agent_profile_item_template", language).format(
        alias=unit.identity.alias or "",
        description=description,
    )


def _text(key: str, language: str | None = None) -> str:
    return get_memory_atom_text(key, language)
