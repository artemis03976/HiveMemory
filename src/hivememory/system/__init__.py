"""
HiveMemory System Layer — 顶层编排门面。

Usage:
    from hivememory.system import HiveMemorySystem
    system = HiveMemorySystem.build()
    await system.start()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.config.app import (
    HiveMemoryConfig,
    load_app_config,
)
from hivememory.core.contracts.subsystem import SubsystemProtocol

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

    system 是依赖图的顶点，只被入口导入；惰性导出让仅需 ``system.config``
    等子模块的调用方（脚本、配置校验）不必加载完整组合根及全部子系统。
    """
    if name == "HiveMemorySystem":
        from hivememory.system.system import HiveMemorySystem

        return HiveMemorySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
