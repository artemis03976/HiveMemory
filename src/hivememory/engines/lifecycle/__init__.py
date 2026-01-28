"""
HiveMemory - 记忆生命周期管理模块 (MemoryLifeCycleManagement)

该模块负责记忆的动态演化、垃圾回收和冷热数据管理。

核心组件:
- VitalityCalculator: 生命力分数计算器
- ReinforcementEngine: 动态强化引擎
- MemoryArchiver: 冷存储管理器
- GarbageCollector: 垃圾回收器
- LifecycleEngine: 统一协调层

对应设计文档: PROJECT.md 第 6 章

状态: Stage 3 实现
作者: HiveMemory Team
版本: 0.2.0
"""

import logging

# 接口
from hivememory.engines.lifecycle.interfaces import (
    BaseMemoryArchiver,
    BaseGarbageCollector,
)

# 类型定义
from hivememory.engines.lifecycle.models import (
    EventType,
    ReinforcementResult,
    MemoryEvent,
    ArchiveStatus,
    ArchiveRecord,
)

# 具体实现 - 生命力计算
from hivememory.engines.lifecycle.vitality import VitalityCalculator

# 具体实现 - 强化引擎
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine

# 具体实现 - 归档器
from hivememory.engines.lifecycle.archiver import (
    FileBasedArchiver,
    create_archiver,
)

# 具体实现 - 垃圾回收器
from hivememory.engines.lifecycle.garbage_collector import (
    PeriodicGarbageCollector,
    ScheduledGarbageCollector,
    create_garbage_collector,
)

# 具体实现 - 生命周期引擎
from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine

logger = logging.getLogger(__name__)


__all__ = [
    # === 接口 ===
    "BaseMemoryArchiver",
    "BaseGarbageCollector",
    # === 类型 ===
    "EventType",
    "ReinforcementResult",
    "MemoryEvent",
    "ArchiveStatus",
    "ArchiveRecord",
    # === 生命力计算 ===
    "VitalityCalculator",
    # === 强化引擎 ===
    "DynamicReinforcementEngine",
    # === 归档器 ===
    "FileBasedArchiver",
    "create_archiver",
    # === 垃圾回收器 ===
    "PeriodicGarbageCollector",
    "ScheduledGarbageCollector",
    "create_garbage_collector",
    # === 生命周期引擎 ===
    "MemoryLifecycleEngine",
]
