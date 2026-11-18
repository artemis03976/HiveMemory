"""AgentProfile → MemoryUnitIR 构建器：只用于可用 agent 列表的菜单项。"""

from __future__ import annotations

from hivememory.core.models import AgentProfile
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)


def build_agent_profile_ir(profile: AgentProfile) -> MemoryUnitIR:
    """菜单项只用 agent_id、title 与 summary；人设只在这个 agent 执行时载入。"""
    return MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="agent_profile", alias=profile.agent_id),
        content=MemoryContentIR(title=profile.title, summary=profile.summary),
        status=MemoryStatusIR(),
    )
