"""
HiveMemory - 生命周期管理器

协调所有生命周期组件，提供统一操作接口。

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.interfaces import BaseGarbageCollector
from hivememory.engines.lifecycle.models import (
    EventType,
    MemoryEvent,
    ReinforcementResult,
)
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.vitality import VitalityCalculator

if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)


class MemoryLifecycleEngine:
    """Coordinate lifecycle scoring, reinforcement events, and garbage collection."""

    def __init__(
        self,
        mid_term: "MidTermMemoryStore",
        vitality_calculator: VitalityCalculator,
        reinforcement_engine: DynamicReinforcementEngine,
        garbage_collector: BaseGarbageCollector,
    ):
        self._mid_term = mid_term
        self.vitality_calculator = vitality_calculator
        self.reinforcement_engine = reinforcement_engine
        self.garbage_collector = garbage_collector
        logger.info("MemoryLifecycleEngine initialized with all components")

    async def refresh_vitality(
        self,
        memory: MemoryAtom,
        *,
        persist: bool = False,
    ) -> float:
        """Refresh a MemoryAtom vitality score in place."""
        vitality = self.vitality_calculator.calculate(memory)
        memory.meta.vitality_score = vitality
        if persist:
            await self._mid_term.upsert(memory)
        return vitality

    async def refresh_vitality_batch(
        self,
        memories: Iterable[MemoryAtom],
        *,
        persist: bool = False,
    ) -> List[Tuple[UUID, float]]:
        """Refresh vitality scores for the caller-provided memory collection."""
        results = []
        for memory in memories:
            vitality = await self.refresh_vitality(memory, persist=persist)
            results.append((memory.id, vitality))
        return results

    async def record_event(self, event: MemoryEvent) -> ReinforcementResult:
        return await self.reinforcement_engine.reinforce(event.memory_id, event)

    async def record_hit(
        self,
        memory_id: UUID,
        source: str = "system",
    ) -> ReinforcementResult:
        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=memory_id,
            source=source,
        )
        return await self.record_event(event)

    async def record_citation(
        self,
        memory_id: UUID,
        source: str = "system",
    ) -> ReinforcementResult:
        event = MemoryEvent(
            event_type=EventType.CITATION,
            memory_id=memory_id,
            source=source,
        )
        return await self.record_event(event)

    async def record_feedback(
        self,
        memory_id: UUID,
        positive: bool,
        source: str = "user",
    ) -> ReinforcementResult:
        event_type = (
            EventType.FEEDBACK_POSITIVE
            if positive
            else EventType.FEEDBACK_NEGATIVE
        )
        event = MemoryEvent(
            event_type=event_type,
            memory_id=memory_id,
            source=source
        )
        return await self.record_event(event)

    async def run_garbage_collection(self, force: bool = False) -> int:
        all_memories = await self._mid_term.scroll(limit=10000)
        await self.refresh_vitality_batch(all_memories, persist=True)
        return await self.garbage_collector.collect(all_memories, force=force)

    async def get_low_vitality_memories(
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
        all_memories = await self._mid_term.scroll(limit=10000)
        refreshed = await self.refresh_vitality_batch(all_memories, persist=False)
        results = [
            (memory_id, vitality)
            for memory_id, vitality in refreshed
            if vitality <= threshold
        ]
        results.sort(key=lambda item: item[1])
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

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 包含各组件统计信息的字典
        """
        stats = {
            "garbage_collector": (
                self.garbage_collector.get_stats()
                if hasattr(self.garbage_collector, "get_stats")
                else {}
            ),
        }

        if hasattr(self.reinforcement_engine, "get_stats"):
            stats["reinforcement"] = self.reinforcement_engine.get_stats()

        return stats


__all__ = [
    "MemoryLifecycleEngine",
]
