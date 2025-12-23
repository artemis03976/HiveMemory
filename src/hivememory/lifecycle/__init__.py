"""
HiveMemory - 记忆生命周期管理模块 (MemoryLifeCycleManagement)

该模块负责记忆的动态演化、垃圾回收和冷热数据管理。

核心组件 (待实现):
- VitalityCalculator: 生命力分数计算器
- ReinforcementEngine: 动态强化引擎
- GarbageCollector: 垃圾回收器
- MemoryArchiver: 冷存储管理器

对应设计文档: PROJECT.md 第 6 章

状态: 骨架接口（Stage 3 实现）
作者: HiveMemory Team
版本: 0.1.0
"""

from hivememory.lifecycle.interfaces import (
    VitalityCalculator,
    ReinforcementEngine,
    MemoryArchiver,
)

__all__ = [
    "VitalityCalculator",
    "ReinforcementEngine",
    "MemoryArchiver",
]
