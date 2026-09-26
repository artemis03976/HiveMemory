"""
HiveMemory System Layer — 顶层编排门面。

Usage:
    from hivememory.system import HiveMemorySystem
    system = HiveMemorySystem.build()
    await system.start()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.system.config import HiveMemoryConfig, load_app_config
from hivememory.system.contracts.subsystem import SubsystemProtocol

if TYPE_CHECKING:
    from hivememory.system.system import HiveMemorySystem

__all__ = [
    "HiveMemorySystem",
    "SubsystemProtocol",
    "HiveMemoryConfig",
    "load_app_config",
]


def __getattr__(name: str) -> Any:
    """惰性导出 ``HiveMemorySystem``（PEP 562）。

    A2 §8 D-2：``workspace.capability`` 依赖 ``system.contracts`` /
    ``system.runtime`` 等子模块，而组合根又装配能力层；包初始化若急切导入
    ``HiveMemorySystem``，会形成 system → assembler → capability → system
    的循环导入。TODO(A5/A6)：能力层对 system 基础设施的依赖方向重新整理后，
    评估恢复急切导出。
    """
    if name == "HiveMemorySystem":
        from hivememory.system.system import HiveMemorySystem

        return HiveMemorySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
