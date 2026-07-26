"""
HiveMemory - 生命力计算器单元测试

测试内容:
- 固有价值权重
- 时间衰减函数
- 访问加成计算
- 边界值处理
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.lifecycle.vitality import (
    VitalityCalculator,
    INTRINSIC_VALUE_WEIGHTS,
)
from hivememory.system.config import VitalityCalculatorConfig


class TestVitalityCalculator:
    """测试生命力分数计算 (三段式 V = V_0·D(t) + A + B)"""

    def setup_method(self):
        """测试初始化: 使用真实配置而非 Mock，避免新字段缺失"""
        self.config = VitalityCalculatorConfig()  # 默认: V_0=100, λ=0.01, access_boost_coef=10.0
        self.calculator = VitalityCalculator(self.config)

    def test_intrinsic_value_weights(self):
        """测试固有价值权重"""
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.CODE_SNIPPET] == 1.0
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.FACT] == 0.9
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.URL_RESOURCE] == 0.8
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.REFLECTION] == 0.7
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.USER_PROFILE] == 0.6
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.WORK_IN_PROGRESS] == 0.5

    def test_calculate_new_memory(self):
        """测试新创建的记忆分数 (三段式: V_0·D(0) + A(0) + B(0) = 100)"""
        memory = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.CODE_SNIPPET,
            access_count=0,
            days_ago=0
        )

        score = self.calculator.calculate(memory)

        # V = V_0·D(0) + 0 + 0 = 100 (V_0 与 confidence 解耦)
        # days_ago=0 仍会有微秒级时间差，D ≈ 0.99999...
        assert score == pytest.approx(100.0, abs=0.01)

    def test_time_decay(self):
        """测试时间衰减"""
        # 新记忆
        memory_new = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=0
        )
        # 100天前的记忆
        memory_old = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=100
        )

        score_new = self.calculator.calculate(memory_new)
        score_old = self.calculator.calculate(memory_old)

        # 旧记忆应该有更低分数
        assert score_old < score_new
        # 在 100 天时，衰减应该约为 37%
        assert score_old < score_new * 0.5

    def test_access_boost(self):
        """测试访问加成 (对数曲线 A = coef·log(1 + n))"""
        # 选 days_ago=30 让 base 足够小，加 A 项后不触发 100 clamp
        memory_no_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=30
        )
        memory_with_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=5,
            days_ago=30
        )

        score_no_access = self.calculator.calculate(memory_no_access)
        score_with_access = self.calculator.calculate(memory_with_access)

        # A(5) = coef · log(6) = 10 · 1.7917 ≈ 17.92
        import math
        expected_diff = self.config.access_boost_coef * math.log(6)
        assert abs((score_with_access - score_no_access) - expected_diff) < 0.5

    def test_access_boost_cap(self):
        """测试访问加成对数饱和 (无硬上限，但增长递减)"""
        # 选 days_ago=200 让 base 足够小，加 A(100) 后不触发 100 clamp
        memory_heavy_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=100,  # 高 access_count
            days_ago=200
        )

        score = self.calculator.calculate(memory_heavy_access)

        # base = V_0 · D(t)；A(100) = coef · log(101) ≈ 46.15
        import math
        fact_intrinsic = self.config.fact_weight
        lambda_eff = self.config.decay_lambda * (2.0 - fact_intrinsic)
        base_score = self.config.base_vitality * math.exp(-lambda_eff * 200)
        expected_access_boost = self.config.access_boost_coef * math.log(101)
        expected_score = base_score + expected_access_boost

        assert abs(score - expected_score) < 0.5
        # 验证对数饱和: access_count=100 的加成不应超过 coef·log(200) (即使再翻倍 access)
        assert expected_access_boost < self.config.access_boost_coef * math.log(200)

    def test_clamping(self):
        """测试分数限制在 [0, 100]"""
        # 极低分数记忆
        memory_low = self._create_memory(
            confidence=0.1,
            memory_type=MemoryType.WORK_IN_PROGRESS,
            access_count=0,
            days_ago=1000
        )

        # 极高分数记忆
        memory_high = self._create_memory(
            confidence=1.0,
            memory_type=MemoryType.CODE_SNIPPET,
            access_count=100,
            days_ago=0
        )

        score_low = self.calculator.calculate(memory_low)
        score_high = self.calculator.calculate(memory_high)

        assert 0 <= score_low <= 100
        assert 0 <= score_high <= 100

    def test_different_memory_types(self):
        """测试不同记忆类型的分数差异"""
        # 相同条件，不同类型
        memory_code = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.CODE_SNIPPET,
            access_count=0,
            days_ago=10
        )
        memory_wip = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.WORK_IN_PROGRESS,
            access_count=0,
            days_ago=10
        )

        score_code = self.calculator.calculate(memory_code)
        score_wip = self.calculator.calculate(memory_wip)

        # CODE_SNIPPET 应该比 WORK_IN_PROGRESS 分数高
        assert score_code > score_wip

    def _create_memory(
        self,
        confidence: float,
        memory_type: MemoryType,
        access_count: int,
        days_ago: int
    ) -> MemoryAtom:
        """创建测试记忆"""
        created_at = datetime.now() - timedelta(days=days_ago)

        return MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=confidence,
                access_count=access_count,
                created_at=created_at,
                updated_at=created_at,
            ),
            index=IndexLayer(
                title="Test Memory",
                summary="Test summary",
                tags=["test"],
                memory_type=memory_type,
            ),
            payload=PayloadLayer(content="Test content"),
        )