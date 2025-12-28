"""
HiveMemory - 强化引擎单元测试

测试内容:
- HIT 事件处理
- CITATION 事件处理
- FEEDBACK 事件处理 (正面/负面)
- 置信度调整
- 事件历史跟踪
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.lifecycle.reinforcement import DynamicReinforcementEngine, DEFAULT_VITALITY_ADJUSTMENTS
from hivememory.lifecycle.types import MemoryEvent, EventType


class TestDynamicReinforcementEngine:
    """测试动态强化引擎"""

    def setup_method(self):
        """测试初始化"""
        self.mock_storage = Mock()
        self.mock_vitality_calc = Mock()

        self.engine = DynamicReinforcementEngine(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality_calc,
            enable_event_history=True,
        )

        # 创建测试记忆
        self.test_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=0.5,  # 50/100
                access_count=5,
            ),
            index=IndexLayer(
                title="Test",
                summary="Test summary",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

    def test_hit_event(self):
        """测试 HIT 事件增加生命力"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0  # 50 + 5

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=self.test_memory.id,
            source="test"
        )

        result = self.engine.reinforce(self.test_memory.id, event)

        assert result.event_type == EventType.HIT
        assert result.new_vitality > result.previous_vitality
        assert self.mock_storage.upsert_memory.called

    def test_citation_resets_decay(self):
        """测试 CITATION 事件重置衰减"""
        # 将 updated_at 设置为过去时间，确保更新后的时间肯定更大
        from datetime import timedelta
        self.test_memory.meta.updated_at -= timedelta(seconds=1)
        original_updated_at = self.test_memory.meta.updated_at

        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 70.0  # 提升效果

        event = MemoryEvent(
            event_type=EventType.CITATION,
            memory_id=self.test_memory.id,
            source="test"
        )

        result = self.engine.reinforce(self.test_memory.id, event)

        assert result.event_type == EventType.CITATION

        # CITATION 应该更新记忆的 updated_at
        updated_memory = self.mock_storage.upsert_memory.call_args[0][0]
        assert updated_memory.meta.updated_at > original_updated_at

    def test_negative_feedback_reduces_confidence(self):
        """测试负面反馈降低置信度"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 25.0  # 降低后

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_NEGATIVE,
            memory_id=self.test_memory.id,
            source="user"
        )

        result = self.engine.reinforce(self.test_memory.id, event)

        assert result.event_type == EventType.FEEDBACK_NEGATIVE
        assert result.new_confidence < result.previous_confidence
        # 应该降低 50%
        assert abs(result.new_confidence - 0.4) < 0.01  # 0.8 * 0.5

    def test_positive_feedback_increases_vitality(self):
        """测试正面反馈增加生命力"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 100.0  # 大幅提升

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_POSITIVE,
            memory_id=self.test_memory.id,
            source="user"
        )

        result = self.engine.reinforce(self.test_memory.id, event)

        assert result.event_type == EventType.FEEDBACK_POSITIVE
        assert result.new_vitality > result.previous_vitality

    def test_memory_not_found(self):
        """测试记忆不存在时抛出异常"""
        self.mock_storage.get_memory.return_value = None

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=uuid4(),
            source="test"
        )

        with pytest.raises(ValueError):
            self.engine.reinforce(uuid4(), event)

    def test_access_count_increments(self):
        """测试访问计数增加"""
        original_count = self.test_memory.meta.access_count

        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=self.test_memory.id,
            source="test"
        )

        self.engine.reinforce(self.test_memory.id, event)

        # 获取更新的记忆
        updated_memory = self.mock_storage.upsert_memory.call_args[0][0]
        assert updated_memory.meta.access_count == original_count + 1

    def test_last_accessed_at_updated(self):
        """测试最后访问时间更新"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=self.test_memory.id,
            source="test"
        )

        self.engine.reinforce(self.test_memory.id, event)

        # 获取更新的记忆
        updated_memory = self.mock_storage.upsert_memory.call_args[0][0]
        assert updated_memory.meta.last_accessed_at is not None

    def test_event_history_tracked(self):
        """测试事件历史跟踪"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=self.test_memory.id,
            source="test"
        )

        self.engine.reinforce(self.test_memory.id, event)

        history = self.engine.get_event_history()
        assert len(history) == 1
        assert history[0].event_type == EventType.HIT

    def test_event_history_filtered_by_memory(self):
        """测试按记忆ID过滤历史"""
        memory1_id = uuid4()
        memory2_id = uuid4()

        # 创建两个记忆
        memory1 = MemoryAtom(
            id=memory1_id,
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=0.5,
            ),
            index=IndexLayer(
                title="Test1",
                summary="Test summary 1",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

        memory2 = MemoryAtom(
            id=memory2_id,
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=0.5,
            ),
            index=IndexLayer(
                title="Test2",
                summary="Test summary 2",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

        self.mock_storage.get_memory.side_effect = [memory1, memory2]
        self.mock_vitality_calc.calculate.return_value = 55.0

        # 记录两个事件
        event1 = MemoryEvent(event_type=EventType.HIT, memory_id=memory1_id, source="test")
        event2 = MemoryEvent(event_type=EventType.HIT, memory_id=memory2_id, source="test")

        self.engine.reinforce(memory1_id, event1)
        self.engine.reinforce(memory2_id, event2)

        # 过滤 memory1 的事件
        history = self.engine.get_event_history(memory_id=memory1_id)
        assert len(history) == 1
        assert history[0].memory_id == memory1_id

    def test_clear_history(self):
        """测试清空历史"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=self.test_memory.id,
            source="test"
        )

        self.engine.reinforce(self.test_memory.id, event)
        assert len(self.engine.get_event_history()) == 1

        self.engine.clear_history()
        assert len(self.engine.get_event_history()) == 0

    def test_get_stats(self):
        """测试获取统计信息"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        # 记录多个事件
        for i in range(3):
            event = MemoryEvent(
                event_type=EventType.HIT,
                memory_id=self.test_memory.id,
                source=f"test{i}"
            )
            self.engine.reinforce(self.test_memory.id, event)

        stats = self.engine.get_stats()
        assert stats["total_events"] == 3
        assert "event_counts" in stats


class TestDefaultVitalityAdjustments:
    """测试默认生命力调整值"""

    def test_adjustments_defined(self):
        """测试所有事件类型都有定义调整值"""
        assert EventType.HIT in DEFAULT_VITALITY_ADJUSTMENTS
        assert EventType.CITATION in DEFAULT_VITALITY_ADJUSTMENTS
        assert EventType.FEEDBACK_POSITIVE in DEFAULT_VITALITY_ADJUSTMENTS
        assert EventType.FEEDBACK_NEGATIVE in DEFAULT_VITALITY_ADJUSTMENTS

    def test_adjustment_values(self):
        """测试调整值符合预期"""
        assert DEFAULT_VITALITY_ADJUSTMENTS[EventType.HIT] == 5.0
        assert DEFAULT_VITALITY_ADJUSTMENTS[EventType.CITATION] == 20.0
        assert DEFAULT_VITALITY_ADJUSTMENTS[EventType.FEEDBACK_POSITIVE] == 50.0
        assert DEFAULT_VITALITY_ADJUSTMENTS[EventType.FEEDBACK_NEGATIVE] == -50.0
