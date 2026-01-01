"""
HiveMemory - Agents 模块

作者: HiveMemory Team
版本: 2.0.0
"""

__all__ = [
    "PatchouliAgent",
    "create_patchouli_agent",
]


def __getattr__(name: str):
    """延迟导入以避免循环依赖"""
    if name == "PatchouliAgent":
        from hivememory.agents.patchouli import PatchouliAgent
        return PatchouliAgent
    elif name == "create_patchouli_agent":
        from hivememory.agents.patchouli import create_patchouli_agent
        return create_patchouli_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
