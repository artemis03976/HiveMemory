"""
HiveMemory - 生命周期管理器

协调所有生命周期组件，提供统一操作接口。

作者: HiveMemory Team
版本: 0.2.0
"""

import logging
from typing import List, Optional, Tuple
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.interfaces import (
    BaseMemoryArchiver,
    BaseGarbageCollector,
)
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.engines.lifecycle.models import (
    MemoryEvent,
    EventType,
    ReinforcementResult,
)
from hivememory.infrastructure.storage import QdrantMemoryStore

logger = logging.getLogger(__name__)


class MemoryLifecycleEngine:
    """
    记忆生命周期管理器

    统一协调所有生命周期组件:
    - VitalityCalculator: 生命力计算
    - ReinforcementEngine: 强化事件处理
    - MemoryArchiver: 冷存储管理
    - GarbageCollector: 垃圾回收

    Examples:
        >>> # 推荐：使用工厂函数
        >>> from hivememory.engines.lifecycle import create_default_lifecycle_manager
        >>> manager = create_default_lifecycle_manager(storage)
        >>>
        >>> # 高级：手动注入组件
        >>> manager = MemoryLifecycleManager(
        ...     storage=storage,
        ...     vitality_calculator=my_calculator,
        ...     reinforcement_engine=my_engine,
        ...     archiver=my_archiver,
        ...     garbage_collector=my_gc,
        ... )
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        vitality_calculator: VitalityCalculator,
        reinforcement_engine: DynamicReinforcementEngine,
        archiver: BaseMemoryArchiver,
        garbage_collector: BaseGarbageCollector,
    ):
        """
        初始化记忆生命周期管理器

        Args:
            storage: 向量存储实例 (QdrantMemoryStore)
            vitality_calculator: 生命力计算器
            reinforcement_engine: 强化引擎
            archiver: 归档器
            garbage_collector: 垃圾回收器

        Note:
            所有组件参数都是必需的。如需使用默认配置创建组件，
            请使用 create_default_lifecycle_manager 工厂函数。
        """
        self.storage = storage
        self.vitality_calculator = vitality_calculator
        self.reinforcement_engine = reinforcement_engine
        self.archiver = archiver
        self.garbage_collector = garbage_collector

        logger.info("MemoryLifecycleManager initialized with all components")

    def calculate_vitality(self, memory_id: UUID) -> float:
        """
        计算并返回生命力分数，同时更新存储中的缓存值

        Args:
            memory_id: 记忆ID

        Returns:
            float: 生命力分数 (0-100)

        Raises:
            ValueError: 记忆不存在
        """
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found")

        vitality = self.vitality_calculator.calculate(memory)

        # 更新存储中的 vitality_score 缓存
        memory.meta.vitality_score = vitality / 100.0
        self.storage.upsert_memory(memory)

        return vitality

    def record_event(self, event: MemoryEvent) -> ReinforcementResult:
        """
        记录生命周期事件

        Args:
            event: 事件对象

        Returns:
            ReinforcementResult: 强化结果

        Raises:
            ValueError: 记忆不存在
        """
        return self.reinforcement_engine.reinforce(event.memory_id, event)

    def record_hit(
        self,
        memory_id: UUID,
        source: str = "system"
    ) -> ReinforcementResult:
        """
        记录检索命中事件 (HIT)

        Args:
            memory_id: 记忆ID
            source: 事件来源

        Returns:
            ReinforcementResult: 强化结果
        """
        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=memory_id,
            source=source
        )
        return self.record_event(event)

    def record_citation(
        self,
        memory_id: UUID,
        source: str = "system"
    ) -> ReinforcementResult:
        """
        记录主动引用事件 (CITATION)

        Args:
            memory_id: 记忆ID
            source: 事件来源

        Returns:
            ReinforcementResult: 强化结果
        """
        event = MemoryEvent(
            event_type=EventType.CITATION,
            memory_id=memory_id,
            source=source
        )
        return self.record_event(event)

    def record_feedback(
        self,
        memory_id: UUID,
        positive: bool,
        source: str = "user"
    ) -> ReinforcementResult:
        """
        记录用户反馈事件

        Args:
            memory_id: 记忆ID
            positive: 是否正面反馈
            source: 事件来源

        Returns:
            ReinforcementResult: 强化结果
        """
        event_type = (
            EventType.FEEDBACK_POSITIVE if positive
            else EventType.FEEDBACK_NEGATIVE
        )
        event = MemoryEvent(
            event_type=event_type,
            memory_id=memory_id,
            source=source
        )
        return self.record_event(event)

    def run_garbage_collection(self, force: bool = False) -> int:
        """
        运行垃圾回收

        Args:
            force: 强制执行

        Returns:
            int: 归档的记忆数量
        """
        return self.garbage_collector.collect(force=force)

    def archive_memory(self, memory_id: UUID) -> None:
        """
        手动归档指定记忆

        Args:
            memory_id: 记忆ID

        Raises:
            ValueError: 记忆不存在
        """
        self.archiver.archive(memory_id)

    def resurrect_memory(self, memory_id: UUID) -> MemoryAtom:
        """
        唤醒归档记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryAtom: 唤醒的记忆

        Raises:
            ValueError: 记忆未归档
        """
        return self.archiver.resurrect(memory_id)

    def get_low_vitality_memories(
        self,
        threshold: float = 20.0,
        limit: int = 100
    ) -> List[Tuple[UUID, float]]:
        """
        获取低于阈值的记忆列表

        Args:
            threshold: 生命力阈值 (0-100)
            limit: 最大返回数量

        Returns:
            List[Tuple[UUID, float]]: (memory_id, vitality) 列表，按生命力升序
        """
        # 获取所有记忆
        all_memories = self.storage.get_all_memories(limit=10000)

        results = []
        for memory in all_memories:
            vitality = self.vitality_calculator.calculate(memory)
            if vitality <= threshold:
                results.append((memory.id, vitality))

        # 按生命力排序 (最低的在前)
        results.sort(key=lambda x: x[1])

        return results[:limit]

    def get_event_history(
        self,
        memory_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[ReinforcementResult]:
        """
        获取事件历史

        Args:
            memory_id: 过滤指定记忆 (None 表示全部)
            limit: 最大返回数量

        Returns:
            List[ReinforcementResult]: 事件历史列表
        """
        if hasattr(self.reinforcement_engine, "get_event_history"):
            return self.reinforcement_engine.get_event_history(memory_id, limit)
        return []

    def get_archived_memories(
        self,
        limit: int = 100,
        vitality_threshold: Optional[float] = None
    ) -> List:
        """
        获取已归档的记忆列表

        Args:
            limit: 最大返回数量
            vitality_threshold: 过滤归档时的生命力阈值

        Returns:
            List[ArchiveRecord]: 归档记录列表
        """
        return self.archiver.list_archived(limit, vitality_threshold)

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 包含各组件统计信息的字典
        """
        stats = {
            "garbage_collector": self.garbage_collector.get_stats() if hasattr(self.garbage_collector, "get_stats") else {},
        }

        if hasattr(self.reinforcement_engine, "get_stats"):
            stats["reinforcement"] = self.reinforcement_engine.get_stats()

        # 添加归档统计
        if hasattr(self.archiver, "_index"):
            stats["archive"] = {
                "total_archived": len(self.archiver._index)
            }

        return stats


__all__ = [
    "MemoryLifecycleEngine",
]
