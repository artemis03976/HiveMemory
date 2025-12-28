"""
HiveMemory - 记忆生命周期管理模块 (MemoryLifeCycleManagement)

该模块负责记忆的动态演化、垃圾回收和冷热数据管理。

核心组件:
- VitalityCalculator: 生命力分数计算器
- ReinforcementEngine: 动态强化引擎
- MemoryArchiver: 冷存储管理器
- GarbageCollector: 垃圾回收器
- LifecycleManager: 统一协调层

对应设计文档: PROJECT.md 第 6 章

状态: Stage 3 实现
作者: HiveMemory Team
版本: 0.1.0
"""

# 接口
from hivememory.lifecycle.interfaces import (
    VitalityCalculator,
    ReinforcementEngine,
    MemoryArchiver,
    GarbageCollector,
    LifecycleManager,
)

# 类型定义
from hivememory.lifecycle.types import (
    EventType,
    ReinforcementResult,
    MemoryEvent,
    ArchiveStatus,
    ArchiveRecord,
)

# 具体实现 - 生命力计算
from hivememory.lifecycle.vitality import (
    StandardVitalityCalculator,
    DecayResetVitalityCalculator,
    INTRINSIC_VALUE_WEIGHTS,
    create_default_vitality_calculator,
)

# 具体实现 - 强化引擎
from hivememory.lifecycle.reinforcement import (
    DynamicReinforcementEngine,
    DEFAULT_VITALITY_ADJUSTMENTS,
    create_default_reinforcement_engine,
)

# 具体实现 - 归档器
from hivememory.lifecycle.archiver import (
    FileBasedMemoryArchiver,
    S3MemoryArchiver,
    create_default_archiver,
)

# 具体实现 - 垃圾回收器
from hivememory.lifecycle.garbage_collector import (
    PeriodicGarbageCollector,
    ScheduledGarbageCollector,
    create_default_garbage_collector,
)

# 具体实现 - 生命周期管理器
from hivememory.lifecycle.orchestrator import (
    MemoryLifecycleManager,
    create_default_lifecycle_manager,
)

__all__ = [
    # === 接口 ===
    "VitalityCalculator",
    "ReinforcementEngine",
    "MemoryArchiver",
    "GarbageCollector",
    "LifecycleManager",

    # === 类型 ===
    "EventType",
    "ReinforcementResult",
    "MemoryEvent",
    "ArchiveStatus",
    "ArchiveRecord",

    # === 生命力计算 ===
    "StandardVitalityCalculator",
    "DecayResetVitalityCalculator",
    "INTRINSIC_VALUE_WEIGHTS",
    "create_default_vitality_calculator",

    # === 强化引擎 ===
    "DynamicReinforcementEngine",
    "DEFAULT_VITALITY_ADJUSTMENTS",
    "create_default_reinforcement_engine",

    # === 归档器 ===
    "FileBasedMemoryArchiver",
    "S3MemoryArchiver",
    "create_default_archiver",

    # === 垃圾回收器 ===
    "PeriodicGarbageCollector",
    "ScheduledGarbageCollector",
    "create_default_garbage_collector",

    # === 生命周期管理器 ===
    "MemoryLifecycleManager",
    "create_default_lifecycle_manager",
]
