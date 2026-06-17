"""
HiveMemory - 动态强化引擎

处理记忆生命周期事件，动态调整生命力分数和置信度。

事件效果:
- HIT: +5 生命力, access_count +1
- CITATION: +20 生命力, 重置时间衰减, access_count +1
- FEEDBACK_POSITIVE: +50 生命力, access_count +1
- FEEDBACK_NEGATIVE: -50 生命力, -50% 置信度

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.models import (
    MemoryEvent,
    EventType,
    ReinforcementResult,
)
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.system.config import ReinforcementEngineConfig

logger = logging.getLogger(__name__)


class DynamicReinforcementEngine:
    """
    动态强化引擎

    处理记忆的生命周期事件，更新生命力分数和置信度。

    工作流程:
        1. 从存储获取记忆
        2. 根据事件类型应用调整
        3. 重新计算生命力分数
        4. 更新访问元信息
        5. 持久化到存储

    Examples:
        >>> engine = DynamicReinforcementEngine(storage, vitality_calculator)
        >>> event = MemoryEvent(event_type=EventType.HIT, memory_id=uuid, source="system")
        >>> result = engine.reinforce(uuid, event)
        >>> print(f"Vitality: {result.previous_vitality:.1f} -> {result.new_vitality:.1f}")
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        config: ReinforcementEngineConfig,
        vitality_calculator: VitalityCalculator,
    ):
        """
        初始化强化引擎

        Args:
            storage: 向量存储实例 (QdrantMemoryStore)
            config: 强化引擎配置
            vitality_calculator: 生命力计算器
        """
        self.storage = storage
        self.config = config
        self._event_history: List[ReinforcementResult] = []
        
        self.vitality_calculator = vitality_calculator

        # 设置生命力调整值
        self.vitality_adjustments = {
            EventType.HIT: self.config.hit_boost,
            EventType.CITATION: self.config.citation_boost,
            EventType.FEEDBACK_POSITIVE: self.config.positive_feedback_boost,
            EventType.FEEDBACK_NEGATIVE: self.config.negative_feedback_penalty,
        }

    async def reinforce(self, memory_id: UUID, event: MemoryEvent) -> ReinforcementResult:
        """
        处理强化事件

        Args:
            memory_id: 目标记忆ID
            event: 事件对象

        Returns:
            ReinforcementResult: 强化结果

        Raises:
            ValueError: 记忆不存在
        """
        # 从存储获取当前记忆
        memory = await self.storage.get_memory(memory_id)
        if memory is None:
            logger.warning(f"Memory not found for reinforcement: {memory_id}")
            raise ValueError(f"Memory {memory_id} not found")

        # 记录当前状态
        previous_vitality = memory.meta.vitality_score
        previous_confidence = memory.meta.confidence_score

        # 应用事件特定的调整
        if event.event_type == EventType.CITATION:
            self._handle_citation(memory)
        elif event.event_type == EventType.FEEDBACK_NEGATIVE:
            self._handle_negative_feedback(memory)

        # 更新访问元信息（在计算生命力之前，以便反映本次访问的影响）
        memory.meta.access_count += 1
        memory.meta.last_accessed_at = datetime.now()
        memory.meta.updated_at = datetime.now()

        base_vitality = self.vitality_calculator.calculate(memory)
        adjustment = self.vitality_adjustments.get(event.event_type, 0.0)
        new_vitality = self._clamp_vitality(base_vitality + adjustment)
        memory.meta.vitality_score = new_vitality

        # 持久化到存储
        await self.storage.upsert_memory(memory)

        # 创建结果
        result = ReinforcementResult(
            memory_id=memory_id,
            previous_vitality=previous_vitality,
            new_vitality=new_vitality,
            previous_confidence=previous_confidence,
            new_confidence=memory.meta.confidence_score,
            event_type=event.event_type,
            timestamp=event.timestamp,
        )

        # 记录事件历史
        if self.config.enable_event_history:
            self._add_to_history(result)

        logger.info(
            f"Reinforcement applied: {memory_id} | "
            f"{event.event_type.value} | "
            f"Vitality: {previous_vitality:.1f} -> {new_vitality:.1f} | "
            f"Confidence: {previous_confidence:.2f} -> {memory.meta.confidence_score:.2f}"
        )

        return result

    def _handle_citation(self, memory: MemoryAtom) -> None:
        # Reset decay before recalculating vitality.
        memory.meta.updated_at = datetime.now()
        logger.debug("Citation handled for %s: decay reset", memory.id)

    def _handle_negative_feedback(self, memory: MemoryAtom) -> None:
        old_confidence = memory.meta.confidence_score
        memory.meta.confidence_score = max(
            0.0,
            memory.meta.confidence_score * self.config.negative_confidence_multiplier
        )

        logger.debug(
            "Negative feedback for %s: confidence %.2f -> %.2f",
            memory.id,
            old_confidence,
            memory.meta.confidence_score,
        )

    def _add_to_history(self, result: ReinforcementResult) -> None:
        """
        添加到事件历史

        Args:
            result: 强化结果
        """
        self._event_history.append(result)

        # 限制历史大小
        if len(self._event_history) > self.config.event_history_limit:
            self._event_history = self._event_history[
                -self.config.event_history_limit:
            ]

    @staticmethod
    def _clamp_vitality(value: float) -> float:
        return max(0.0, min(100.0, value))

    def get_event_history(
        self,
        memory_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[ReinforcementResult]:
        """
        获取事件历史

        Args:
            memory_id: 过滤指定记忆的事件 (None 表示全部)
            limit: 最大返回数量

        Returns:
            List[ReinforcementResult]: 事件历史列表，最新的在前
        """
        history = self._event_history

        if memory_id is not None:
            history = [r for r in history if r.memory_id == memory_id]

        # 按时间倒序排序
        history = sorted(history, key=lambda x: x.timestamp, reverse=True)

        return history[:limit]

    def clear_history(self) -> None:
        """清空事件历史 (用于测试或维护)"""
        self._event_history.clear()
        logger.info("Event history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        event_counts = {}
        for result in self._event_history:
            event_type = result.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self._event_history),
            "event_counts": event_counts,
            "history_limit": self.config.event_history_limit,
        }


__all__ = [
    "DynamicReinforcementEngine",
]
