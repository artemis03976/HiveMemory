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
from unittest.mock import Mock
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.lifecycle.vitality import (
    StandardVitalityCalculator,
    INTRINSIC_VALUE_WEIGHTS,
)


class TestVitalityCalculator:
    """测试生命力分数计算"""

    def setup_method(self):
        """测试初始化"""
        self.config = Mock()
        self.config.decay_lambda = 0.01
        self.config.code_snippet_weight = 1.0
        self.config.fact_weight = 0.9
        self.config.url_resource_weight = 0.8
        self.config.reflection_weight = 0.7
        self.config.user_profile_weight = 0.6
        self.config.work_in_progress_weight = 0.5
        self.config.default_weight = 0.5
        self.config.points_per_access = 2.0
        self.config.max_access_boost = 20.0
        self.calculator = StandardVitalityCalculator(self.config)

    def test_intrinsic_value_weights(self):
        """测试固有价值权重"""
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.CODE_SNIPPET] == 1.0
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.FACT] == 0.9
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.URL_RESOURCE] == 0.8
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.REFLECTION] == 0.7
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.USER_PROFILE] == 0.6
        assert INTRINSIC_VALUE_WEIGHTS[MemoryType.WORK_IN_PROGRESS] == 0.5

    def test_calculate_new_memory(self):
        """测试新创建的记忆分数"""
        memory = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.CODE_SNIPPET,
            access_count=0,
            days_ago=0
        )

        score = self.calculator.calculate(memory)

        # V = (0.9 * 1.0) * 1.0 * 100 + 0 = 90
        assert 85 <= score <= 95

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
        """测试访问加成"""
        memory_no_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=10
        )
        memory_with_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=5,
            days_ago=10
        )

        score_no_access = self.calculator.calculate(memory_no_access)
        score_with_access = self.calculator.calculate(memory_with_access)

        # 5次访问 = 10点加成 (5 * 2)
        assert 9 <= score_with_access - score_no_access <= 11

    def test_access_boost_cap(self):
        """测试访问加成封顶"""
        memory_heavy_access = self._create_memory(
            confidence=0.8,
            memory_type=MemoryType.FACT,
            access_count=100,  # 远超封顶值
            days_ago=10
        )

        score = self.calculator.calculate(memory_heavy_access)

        # 访问加成应被限制在 20
        base_score = (0.8 * 0.9) * self.calculator._calculate_decay(10) * 100
        assert score <= base_score + 20.1  # 小误差容忍

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


class TestDecayResetVitalityCalculator:
    """测试支持衰减重置的生命力计算器"""

    def setup_method(self):
        """测试初始化"""
        self.config = Mock()
        self.config.decay_lambda = 0.01
        self.config.code_snippet_weight = 1.0
        self.config.fact_weight = 0.9
        self.config.url_resource_weight = 0.8
        self.config.reflection_weight = 0.7
        self.config.user_profile_weight = 0.6
        self.config.work_in_progress_weight = 0.5
        self.config.default_weight = 0.5
        self.config.points_per_access = 2.0
        self.config.max_access_boost = 20.0
        from hivememory.lifecycle.vitality import DecayResetVitalityCalculator
        self.calculator = DecayResetVitalityCalculator(self.config)

    def test_citation_resets_decay(self):
        """测试引用重置衰减"""
        # 创建旧记忆
        old_memory = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=100
        )

        # 不标记引用时的分数
        score_without_citation = self.calculator.calculate(old_memory)

        # 标记为引用
        self.calculator.mark_cited(old_memory.id)

        # 标记引用后的分数应该更高
        score_with_citation = self.calculator.calculate(old_memory)

        assert score_with_citation > score_without_citation

    def test_citation_mark_consumed(self):
        """测试引用标记是一次性的"""
        memory = self._create_memory(
            confidence=0.9,
            memory_type=MemoryType.FACT,
            access_count=0,
            days_ago=100
        )

        # 标记引用
        self.calculator.mark_cited(memory.id)

        # 第一次计算应该使用重置的衰减
        score1 = self.calculator.calculate(memory)

        # 第二次计算不再使用重置的衰减
        score2 = self.calculator.calculate(memory)

        assert score2 < score1  # 第二次没有重置效果

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
