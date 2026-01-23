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
版本: 0.2.0
"""

import logging
from typing import Optional, TYPE_CHECKING
from hivememory.infrastructure.storage import QdrantMemoryStore

# 接口
from hivememory.engines.lifecycle.interfaces import (
    VitalityCalculator,
    ReinforcementEngine,
    MemoryArchiver,
    GarbageCollector,
    LifecycleManager,
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
from hivememory.engines.lifecycle.vitality import (
    StandardVitalityCalculator,
    DecayResetVitalityCalculator,
    INTRINSIC_VALUE_WEIGHTS,
    create_default_vitality_calculator,
)

# 具体实现 - 强化引擎
from hivememory.engines.lifecycle.reinforcement import (
    DynamicReinforcementEngine,
    DEFAULT_VITALITY_ADJUSTMENTS,
    create_default_reinforcement_engine,
)

# 具体实现 - 归档器
from hivememory.engines.lifecycle.archiver import (
    FileBasedMemoryArchiver,
    S3MemoryArchiver,
    create_default_archiver,
)

# 具体实现 - 垃圾回收器
from hivememory.engines.lifecycle.garbage_collector import (
    PeriodicGarbageCollector,
    ScheduledGarbageCollector,
    create_default_garbage_collector,
)

# 具体实现 - 生命周期管理器
from hivememory.engines.lifecycle.orchestrator import (
    MemoryLifecycleManager,
    create_default_lifecycle_manager,
)

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryLifecycleConfig

logger = logging.getLogger(__name__)


def create_lifecycle_manager_from_config(
    storage: QdrantMemoryStore,
    config: Optional["MemoryLifecycleConfig"] = None,
) -> MemoryLifecycleManager:
    """
    根据配置创建记忆生命周期管理器

    根据 config 自动创建并配置所有子组件：
    - VitalityCalculator: 生命力计算器
    - ReinforcementEngine: 强化引擎
    - MemoryArchiver: 归档器
    - GarbageCollector: 垃圾回收器

    Args:
        storage: QdrantMemoryStore 实例
        config: 记忆生命周期配置（可选，使用默认配置）

    Returns:
        MemoryLifecycleManager: 记忆生命周期管理器实例

    Examples:
        >>> from hivememory.lifecycle import create_lifecycle_manager_from_config
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>> manager = create_lifecycle_manager_from_config(storage=storage)
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.patchouli.config import MemoryLifecycleConfig
        >>> config = MemoryLifecycleConfig()
        >>> config.gc_enable_schedule = True
        >>> config.gc_interval_hours = 12
        >>> manager = create_lifecycle_manager_from_config(storage=storage, config=config)

    Note:
        这是一个与 create_default_lifecycle_manager 并存的工厂函数。
        create_default_lifecycle_manager 接受具体的参数，而此函数接受配置对象。
        未来版本可能会合并这两个函数。
    """
    if config is None:
        from hivememory.patchouli.config import MemoryLifecycleConfig
        config = MemoryLifecycleConfig()

    # 使用工厂函数创建管理器，传递完整配置
    manager = create_default_lifecycle_manager(
        storage=storage,
        config=config,
    )

    logger.info("MemoryLifecycleManager created from config")
    return manager


__all__ = [
    # === 接口 ===
    "VitalityCalculator",
    "ReinforcementEngine",
    "MemoryArchiver",
    "GarbageCollector",
    "MemoryLifecycleManager",

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
    "create_lifecycle_manager_from_config",
]
