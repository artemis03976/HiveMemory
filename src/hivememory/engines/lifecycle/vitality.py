"""
HiveMemory - 生命力分数计算器

实现记忆生命力分数的计算逻辑 (三段式语义)。

公式: V(t) = V_0 · D(t) + A(access) + B(events)
- V_0   = base_vitality                         # 初始强度 (固定高值，与 confidence 解耦)
- D(t)  = exp(-λ_eff · days_since_update)        # 时间衰减 (艾宾浩斯遗忘曲线)
         λ_eff = λ · (2 - I),  I 为记忆类型固有价值
- A     = access_boost_coef · log(1 + access_count)  # 访问加成 (对数曲线，自然饱和)
- B     = memory.meta.event_vitality_boost       # 事件累积加成 (HIT/CITATION/FEEDBACK 累积)

设计原则:
    - 记忆最初总是高值 (V_0): 新记忆 vitality = V_0 + 0 + 0 = base_vitality
    - 时间衰退由 D(t) 主导，高价值记忆衰退更慢 (λ_eff 调制)
    - 命中/反馈事件通过 B 项累积，不会因公式重算而丢失
    - confidence 影响"是否容易遗忘" (TODO: 未来可纳入 λ_eff 调制)，而非压低起点

作者: HiveMemory Team
版本: 0.2.0
"""

import math
from datetime import datetime

from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.system.config import VitalityCalculatorConfig


class VitalityCalculator:
    """
    标准生命力分数计算器 (三段式)

    实现公式: V(t) = V_0 · D(t) + A(access) + B(events)

    计算步骤:
        1. 取固定初始强度 V_0 = config.base_vitality
        2. 取固有价值 I (按记忆类型)，计算 λ_eff = λ · (2 - I)
        3. 计算 D(t) = exp(-λ_eff · days_since_update)
        4. 计算访问加成 A = access_boost_coef · log(1 + access_count)
        5. 取事件累积加成 B = memory.meta.event_vitality_boost
        6. V = V_0 · D(t) + A + B，并 clamp 到 [0, 100]

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

        # 构建固有价值权重字典 (I 作为抗衰减调制因子)
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
        # 组件 V_0: 初始强度 (固定高值，与 confidence 解耦)
        v0 = self.config.base_vitality

        # 组件 D(t): 时间衰减 (λ_eff 由记忆类型调制)
        intrinsic_value = self._intrinsic_weights.get(
            memory.index.memory_type,
            self.config.default_weight,
        )
        days_since_update = self._days_since(memory.meta.updated_at)
        decay_factor = self._calculate_decay(days_since_update, intrinsic_value)

        # 组件 A: 访问加成 (对数曲线，自然饱和)
        access_boost = self._calculate_access_boost(memory.meta.access_count)

        # 组件 B: 事件累积加成 (单独存储，由强化引擎维护)
        event_boost = memory.meta.event_vitality_boost

        # 最终公式: V = V_0 · D(t) + A + B
        vitality = (v0 * decay_factor) + access_boost + event_boost

        # 限制在 [0, 100] 范围内
        return max(0.0, min(100.0, vitality))

    def _days_since(self, date: datetime) -> float:
        """
        计算距离指定日期的天数

        Args:
            date: 目标日期

        Returns:
            float: 距今天数 (可以是小数，非负)
        """
        delta = datetime.now() - date
        return max(0.0, delta.total_seconds() / 86400.0)

    def _calculate_decay(self, days: float, intrinsic_value: float) -> float:
        """
        计算时间衰减因子

        公式: D(t) = exp(-λ_eff · t),  λ_eff = λ · (2 - I)

        - t=0 时: D(0) = 1.0 (无衰减)
        - I=1.0 (CODE_SNIPPET): λ_eff = λ (衰减最慢)
        - I=0.5 (WORK_IN_PROGRESS): λ_eff = 1.5λ (衰减更快)
        - λ=0.01, t=100 天, I=1.0: D(100) ≈ 0.37

        Args:
            days: 距离更新的天数
            intrinsic_value: 记忆类型固有价值 (用于调制衰减速率)

        Returns:
            float: 衰减因子 (0-1)
        """
        decay_lambda = self.config.decay_lambda
        lambda_eff = decay_lambda * (2.0 - intrinsic_value)
        return math.exp(-lambda_eff * days)

    def _calculate_access_boost(self, access_count: int) -> float:
        """
        计算访问加成 (对数曲线)

        公式: A = access_boost_coef · log(1 + access_count)

        - access_count=0:  A = 0
        - access_count=1:  A = coef · log(2)  ≈ coef · 0.693
        - access_count=10: A = coef · log(11) ≈ coef · 2.398
        - access_count=100: A = coef · log(101) ≈ coef · 4.615

        对数曲线自然饱和，无需硬上限；高频命中仍保持微弱递增梯度。

        Args:
            access_count: 访问次数

        Returns:
            float: 访问加成分数 (≥0)
        """
        if access_count <= 0:
            return 0.0
        return self.config.access_boost_coef * math.log(1.0 + access_count)


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