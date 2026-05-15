"""
HiveMemory System Layer — 顶层编排门面。

Usage:
    from hivememory.system import HiveMemorySystem
    system = HiveMemorySystem.build()
    await system.start()
"""

from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.system import HiveMemorySystem

__all__ = [
    "HiveMemorySystem",
    "SubsystemProtocol",
]
