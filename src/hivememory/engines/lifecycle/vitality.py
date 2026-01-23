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
from typing import Dict, Optional, TYPE_CHECKING

from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.engines.lifecycle.interfaces import VitalityCalculator

if TYPE_CHECKING:
    from hivememory.patchouli.config import VitalityCalculatorConfig

logger = logging.getLogger(__name__)


# 固有价值权重 - 基于记忆类型 (默认值，可被配置覆盖)
# 代码片段 > 事实 > URL资源 > 反思 > 用户画像 > 进行中
INTRINSIC_VALUE_WEIGHTS: Dict[MemoryType, float] = {
    MemoryType.CODE_SNIPPET: 1.0,
    MemoryType.FACT: 0.9,
    MemoryType.URL_RESOURCE: 0.8,
    MemoryType.REFLECTION: 0.7,
    MemoryType.USER_PROFILE: 0.6,
    MemoryType.WORK_IN_PROGRESS: 0.5,
}


class StandardVitalityCalculator(VitalityCalculator):
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

    def __init__(self, config: Optional["VitalityCalculatorConfig"] = None):
        """
        初始化计算器

        Args:
            config: 生命力计算器配置对象 (可选，使用默认配置如果未提供)
        """
        self.config = config
        if config is None:
            from hivememory.patchouli.config import VitalityCalculatorConfig
            self.config = VitalityCalculatorConfig()

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


class DecayResetVitalityCalculator(StandardVitalityCalculator):
    """
    支持衰减重置的生命力计算器

    当记忆被引用 (CITATION) 时，将重置时间衰减，
    给予记忆临时的生命力提升。

    这种设计使得被引用的"旧知识"能够重新获得高生命力。
    """

    # 引用后重置的天数
    _CITATION_RESET_DAYS = 30

    def __init__(self, config: Optional["VitalityCalculatorConfig"] = None):
        """
        初始化计算器

        Args:
            config: 生命周期配置对象
        """
        super().__init__(config)
        self._citation_memory_ids = set()

    def mark_cited(self, memory_id: "UUID") -> None:
        """
        标记记忆为已引用

        下次计算时，该记忆的时间衰减将被重置。

        Args:
            memory_id: 记忆ID
        """
        self._citation_memory_ids.add(memory_id)
        logger.debug(f"Marked memory {memory_id} as cited (decay reset pending)")

    def calculate(self, memory: MemoryAtom) -> float:
        """
        计算生命力分数 (支持引用衰减重置)

        Args:
            memory: 记忆原子

        Returns:
            float: 生命力分数 (0-100)
        """
        # 如果记忆被标记为引用，调整有效更新时间
        if memory.id in self._citation_memory_ids:
            # 创建修改后的记忆对象，使用当前时间作为更新时间
            # 这将重置时间衰减
            from copy import deepcopy

            modified = deepcopy(memory)
            modified.meta.updated_at = datetime.now()

            result = super().calculate(modified)

            # 移除标记 (一次性效果)
            self._citation_memory_ids.discard(memory.id)

            logger.debug(
                f"Calculated vitality with citation reset for {memory.id}: {result:.1f}"
            )
            return result

        return super().calculate(memory)


def create_default_vitality_calculator(
    config: Optional["VitalityCalculatorConfig"] = None
) -> VitalityCalculator:
    """
    创建默认生命力计算器

    Args:
        config: 生命周期配置 (可选)

    Returns:
        VitalityCalculator: 计算器实例
    """
    return StandardVitalityCalculator(config)


__all__ = [
    "StandardVitalityCalculator",
    "DecayResetVitalityCalculator",
    "INTRINSIC_VALUE_WEIGHTS",
    "create_default_vitality_calculator",
]
