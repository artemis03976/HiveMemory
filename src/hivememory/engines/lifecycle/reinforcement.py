"""
HiveMemory - 动态强化引擎

处理记忆生命周期事件，动态调整生命力分数和置信度。

事件效果 (改造成三段式语义后的新契约):
- HIT:              event_vitality_boost += 5,  access_count += 1,  不重置衰减
- CITATION:         event_vitality_boost += 20, access_count += 1,  重置时间衰减 (updated_at = now)
- FEEDBACK_POSITIVE: event_vitality_boost += 50, access_count += 1, 不重置衰减
- FEEDBACK_NEGATIVE: event_vitality_boost += -50, confidence ×0.5,  不重置衰减

事件加成不再直接加减最终 vitality_score，而是累加进 event_vitality_boost (B 项)。
最终 vitality_score 由 VitalityCalculator 重算时统一合并: V = V_0·D(t) + A + B。

HIT 不重置 updated_at —— 让时间衰减在遗忘曲线上持续作用，符合艾宾浩斯语义；
只有 CITATION (主动复习) 才重置 updated_at，对应"主动回忆重置遗忘曲线"。

作者: HiveMemory Team
版本: 0.2.0
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
from hivememory.system.config import ReinforcementEngineConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

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
        mid_term: "MidTermMemoryStore",
        config: ReinforcementEngineConfig,
        vitality_calculator: VitalityCalculator,
    ):
        self._mid_term = mid_term
        self.config = config
        self._event_history: List[ReinforcementResult] = []
        self.vitality_calculator = vitality_calculator
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
        memory = await self._mid_term.get(memory_id)
        if memory is None:
            logger.warning(f"Memory not found for reinforcement: {memory_id}")
            raise ValueError(f"Memory {memory_id} not found")

        # 记录当前状态
        previous_vitality = memory.meta.vitality_score
        previous_confidence = memory.meta.confidence_score

        # 应用事件特定的调整 (CITATION 重置 updated_at，FEEDBACK_NEGATIVE 调整 confidence)
        if event.event_type == EventType.CITATION:
            self._handle_citation(memory)
        elif event.event_type == EventType.FEEDBACK_NEGATIVE:
            self._handle_negative_feedback(memory)

        # 事件加成累加进 B 项 (event_vitality_boost)，不直接改 vitality_score
        # 最终 vitality_score 由 VitalityCalculator 在重算时统一合并: V = V_0·D(t) + A + B
        adjustment = self.vitality_adjustments.get(event.event_type, 0.0)
        memory.meta.event_vitality_boost = max(
            -100.0,
            min(100.0, memory.meta.event_vitality_boost + adjustment),
        )

        # 更新访问元信息
        # 注意: 只更新 last_accessed_at；updated_at 仅在 CITATION 主动复习时重置
        # (上面 _handle_citation 已处理)。HIT 不重置 updated_at，让遗忘曲线持续作用。
        memory.meta.access_count += 1
        memory.meta.last_accessed_at = datetime.now()

        # 由 VitalityCalculator 统一重算 (包含 V_0/D(t)/A/B 三段)
        new_vitality = self._clamp_vitality(self.vitality_calculator.calculate(memory))
        memory.meta.vitality_score = new_vitality

        # 持久化到存储
        await self._mid_term.upsert(memory)

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
