"""
HiveMemory System Layer — 顶层编排门面。

Usage:
    from hivememory.system import HiveMemorySystem, SystemBootstrap
    system = SystemBootstrap.build()
    await system.start()
"""

from hivememory.system.bootstrap import SystemBootstrap
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.system import HiveMemorySystem

__all__ = [
    "HiveMemorySystem",
    "SubsystemProtocol",
    "SystemBootstrap",
]
