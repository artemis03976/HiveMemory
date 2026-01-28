"""
HiveMemory - 生命力分数计算器

实现记忆生命力分数的计算逻辑。

公式: V = (C × I) × D(t) + A
- C = Confidence Score (置信度)
- I = Intrinsic Value (固有价值，基于记忆类型)
- D(t) = Decay Function (时间衰减函数)
- A = Access Boost (访问加成)

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
import math
from datetime import datetime
from typing import List, Tuple, TYPE_CHECKING
from uuid import UUID

from hivememory.patchouli.config import VitalityCalculatorConfig
from hivememory.core.models import MemoryAtom, MemoryType

if TYPE_CHECKING:
    from hivememory.infrastructure.storage import QdrantMemoryStore

logger = logging.getLogger(__name__)


class VitalityCalculator:
    """
    标准生命力分数计算器

    实现公式: V = (C × I) × D(t) + A

    计算步骤:
        1. 获取置信度 C (0-1)
        2. 获取固有价值 I (0-1, 基于记忆类型)
        3. 计算时间衰减 D(t) = exp(-λ × days_since_update)
        4. 计算访问加成 A = min(max_boost, access_count × points_per_access)
        5. 计算最终分数 V = (C × I) × D(t) × 100 + A

    Examples:
        >>> from hivememory.lifecycle.vitality import create_default_vitality_calculator
        >>> calculator = create_default_vitality_calculator()
        >>> score = calculator.calculate(memory)
    """

    def __init__(self, config: VitalityCalculatorConfig):
        """
        初始化计算器

        Args:
            config: 生命力计算器配置对象
        """
        self.config = config

        # 构建固有价值权重字典
        self._intrinsic_weights = {
            MemoryType.CODE_SNIPPET: self.config.code_snippet_weight,
            MemoryType.FACT: self.config.fact_weight,
            MemoryType.URL_RESOURCE: self.config.url_resource_weight,
            MemoryType.REFLECTION: self.config.reflection_weight,
            MemoryType.USER_PROFILE: self.config.user_profile_weight,
            MemoryType.WORK_IN_PROGRESS: self.config.work_in_progress_weight,
        }

    def calculate(self, memory: MemoryAtom) -> float:
        """
        计算生命力分数

        Args:
            memory: 记忆原子

        Returns:
            float: 生命力分数 (0-100)
        """
        # 组件 C: 置信度分数 (0-1)
        confidence = memory.meta.confidence_score

        # 组件 I: 固有价值 (0-1, 基于记忆类型)
        intrinsic_value = self._intrinsic_weights.get(
            memory.index.memory_type,
            self.config.default_weight
        )

        # 基础分数: C × I (0-1 范围)
        base_score = confidence * intrinsic_value

        # 组件 D(t): 时间衰减
        days_since_update = self._days_since(memory.meta.updated_at)
        decay_factor = self._calculate_decay(days_since_update)

        # 组件 A: 访问加成
        access_boost = self._calculate_access_boost(memory.meta.access_count)

        # 最终公式: V = (C × I) × D(t) × 100 + A
        # 乘以 100 将结果映射到 0-100 范围
        vitality = (base_score * decay_factor * 100.0) + access_boost

        # 限制在 [0, 100] 范围内
        return max(0.0, min(100.0, vitality))

    def _days_since(self, date: datetime) -> float:
        """
        计算距离指定日期的天数

        Args:
            date: 目标日期

        Returns:
            float: 距今天数 (可以是小数)
        """
        delta = datetime.now() - date
        return delta.total_seconds() / 86400.0  # 秒转天

    def _calculate_decay(self, days: float) -> float:
        """
        计算时间衰减因子

        公式: D(t) = exp(-λ × t)

        - t=0 时: D(0) = 1.0 (无衰减)
        - λ=0.01, t=100 天时: D(100) ≈ 0.37

        Args:
            days: 距离更新的天数

        Returns:
            float: 衰减因子 (0-1)
        """
        # 使用配置中的衰减系数
        decay_lambda = self.config.decay_lambda
        return math.exp(-decay_lambda * days)

    def _calculate_access_boost(self, access_count: int) -> float:
        """
        计算访问加成

        公式: A = min(max_boost, access_count × points_per_access)

        Args:
            access_count: 访问次数

        Returns:
            float: 访问加成分数 (0-max_boost)
        """
        raw_boost = access_count * self.config.points_per_access
        return min(self.config.max_access_boost, raw_boost)

    def refresh_batch(
        self,
        memories: List[MemoryAtom],
        storage: "QdrantMemoryStore"
    ) -> List[Tuple[UUID, float]]:
        """
        批量刷新记忆的生命力分数

        计算每个记忆的当前生命力并更新存储中的缓存值。

        Args:
            memories: 待刷新的记忆列表
            storage: 存储实例，用于持久化更新

        Returns:
            List[Tuple[UUID, float]]: (memory_id, new_vitality) 列表
        """
        results = []

        for memory in memories:
            new_vitality = self.calculate(memory)
            memory.meta.vitality_score = new_vitality
            storage.upsert_memory(memory)
            results.append((memory.id, new_vitality))

        logger.info(f"Batch refreshed vitality for {len(memories)} memories")
        return results


# 固有价值权重常量 (供测试使用)
INTRINSIC_VALUE_WEIGHTS = {
    MemoryType.CODE_SNIPPET: 1.0,
    MemoryType.FACT: 0.9,
    MemoryType.URL_RESOURCE: 0.8,
    MemoryType.REFLECTION: 0.7,
    MemoryType.USER_PROFILE: 0.6,
    MemoryType.WORK_IN_PROGRESS: 0.5,
}


__all__ = [
    "VitalityCalculator",
    "INTRINSIC_VALUE_WEIGHTS",
]
