"""
MemoryLifecycleEngine 单元测试

测试覆盖:
- calculate_vitality: 正常计算 / 记忆不存在
- record_hit / record_citation / record_feedback: 事件构建与委托
- run_garbage_collection / archive / resurrect: 委托调用
- get_low_vitality_memories: 过滤 + 排序 + limit
- get_event_history: 有/无该方法
- get_stats: 聚合各组件统计
"""

import pytest
from unittest.mock import Mock, MagicMock, PropertyMock
from uuid import uuid4

from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
from hivememory.engines.lifecycle.models import EventType, ReinforcementResult, MemoryEvent
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType


def _make_memory(vitality_score=0.5) -> MemoryAtom:
    """辅助: 构建测试用 MemoryAtom"""
    mem = MemoryAtom(
        meta=MetaData(
            source_agent_id="a1", user_id="u1", session_id="s1",
            vitality_score=vitality_score,
        ),
        index=IndexLayer(title="测试", summary="这是一段足够长的测试摘要用于通过验证", tags=["t"], memory_type=MemoryType.FACT),
        payload=PayloadLayer(content="内容"),
    )
    return mem


class TestLifecycleEngineVitality:
    """生命力计算测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_vitality = Mock()
        self.mock_reinforcement = Mock()
        self.mock_archiver = Mock()
        self.mock_gc = Mock()
        self.engine = MemoryLifecycleEngine(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality,
            reinforcement_engine=self.mock_reinforcement,
            archiver=self.mock_archiver,
            garbage_collector=self.mock_gc,
        )

    def test_calculate_vitality_success(self):
        """正常计算生命力并更新存储"""
        mem = _make_memory()
        mid = mem.id
        self.mock_storage.get_memory.return_value = mem
        self.mock_vitality.calculate.return_value = 75.0

        result = self.engine.calculate_vitality(mid)

        assert result == 75.0
        assert mem.meta.vitality_score == pytest.approx(0.75)
        self.mock_storage.upsert_memory.assert_called_once_with(mem)

    def test_calculate_vitality_not_found(self):
        """记忆不存在时抛 ValueError"""
        self.mock_storage.get_memory.return_value = None

        with pytest.raises(ValueError, match="not found"):
            self.engine.calculate_vitality(uuid4())


class TestLifecycleEngineEvents:
    """事件记录测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_vitality = Mock()
        self.mock_reinforcement = Mock()
        self.mock_archiver = Mock()
        self.mock_gc = Mock()
        self.engine = MemoryLifecycleEngine(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality,
            reinforcement_engine=self.mock_reinforcement,
            archiver=self.mock_archiver,
            garbage_collector=self.mock_gc,
        )
        self.mock_reinforcement.reinforce.return_value = Mock(spec=ReinforcementResult)

    def test_record_hit(self):
        """HIT 事件正确构建并委托"""
        mid = uuid4()
        self.engine.record_hit(mid, source="retrieval")

        call_args = self.mock_reinforcement.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.HIT
        assert event.memory_id == mid
        assert event.source == "retrieval"

    def test_record_citation(self):
        """CITATION 事件正确构建并委托"""
        mid = uuid4()
        self.engine.record_citation(mid, source="agent")

        call_args = self.mock_reinforcement.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.CITATION
        assert event.source == "agent"

    def test_record_feedback_positive(self):
        """正面反馈事件"""
        mid = uuid4()
        self.engine.record_feedback(mid, positive=True)

        call_args = self.mock_reinforcement.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.FEEDBACK_POSITIVE

    def test_record_feedback_negative(self):
        """负面反馈事件"""
        mid = uuid4()
        self.engine.record_feedback(mid, positive=False)

        call_args = self.mock_reinforcement.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.FEEDBACK_NEGATIVE

    def test_record_event_delegates(self):
        """record_event 委托给 reinforcement_engine"""
        mid = uuid4()
        event = MemoryEvent(event_type=EventType.HIT, memory_id=mid, source="test")
        self.engine.record_event(event)

        self.mock_reinforcement.reinforce.assert_called_once_with(mid, event)


class TestLifecycleEngineDelegation:
    """委托调用测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_vitality = Mock()
        self.mock_reinforcement = Mock()
        self.mock_archiver = Mock()
        self.mock_gc = Mock()
        self.engine = MemoryLifecycleEngine(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality,
            reinforcement_engine=self.mock_reinforcement,
            archiver=self.mock_archiver,
            garbage_collector=self.mock_gc,
        )

    def test_run_garbage_collection(self):
        """委托给 garbage_collector"""
        self.mock_gc.collect.return_value = 3
        result = self.engine.run_garbage_collection(force=True)

        self.mock_gc.collect.assert_called_once_with(force=True)
        assert result == 3

    def test_archive_memory(self):
        """委托给 archiver"""
        mid = uuid4()
        self.engine.archive_memory(mid)
        self.mock_archiver.archive.assert_called_once_with(mid)

    def test_resurrect_memory(self):
        """委托给 archiver"""
        mid = uuid4()
        mem = _make_memory()
        self.mock_archiver.resurrect.return_value = mem

        result = self.engine.resurrect_memory(mid)

        self.mock_archiver.resurrect.assert_called_once_with(mid)
        assert result is mem


class TestLifecycleEngineQueries:
    """查询方法测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_vitality = Mock()
        self.mock_reinforcement = Mock()
        self.mock_archiver = Mock()
        self.mock_gc = Mock()
        self.engine = MemoryLifecycleEngine(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality,
            reinforcement_engine=self.mock_reinforcement,
            archiver=self.mock_archiver,
            garbage_collector=self.mock_gc,
        )

    def test_get_low_vitality_memories(self):
        """过滤低生命力记忆并按升序排序"""
        m1 = _make_memory()
        m2 = _make_memory()
        m3 = _make_memory()
        self.mock_storage.get_all_memories.return_value = [m1, m2, m3]
        # m1=10, m2=50, m3=5
        self.mock_vitality.calculate.side_effect = [10.0, 50.0, 5.0]

        results = self.engine.get_low_vitality_memories(threshold=20.0)

        assert len(results) == 2
        # 按升序: m3(5) 在前, m1(10) 在后
        assert results[0][1] == 5.0
        assert results[1][1] == 10.0

    def test_get_low_vitality_memories_with_limit(self):
        """limit 限制返回数量"""
        m1 = _make_memory()
        m2 = _make_memory()
        self.mock_storage.get_all_memories.return_value = [m1, m2]
        self.mock_vitality.calculate.side_effect = [5.0, 10.0]

        results = self.engine.get_low_vitality_memories(threshold=20.0, limit=1)

        assert len(results) == 1

    def test_get_low_vitality_memories_none_below_threshold(self):
        """所有记忆都高于阈值时返回空"""
        m1 = _make_memory()
        self.mock_storage.get_all_memories.return_value = [m1]
        self.mock_vitality.calculate.return_value = 90.0

        results = self.engine.get_low_vitality_memories(threshold=20.0)

        assert results == []

    def test_get_event_history_supported(self):
        """reinforcement_engine 有 get_event_history 方法时正常返回"""
        mock_history = [Mock(spec=ReinforcementResult)]
        self.mock_reinforcement.get_event_history = Mock(return_value=mock_history)

        results = self.engine.get_event_history(limit=50)

        self.mock_reinforcement.get_event_history.assert_called_once_with(None, 50)
        assert results == mock_history

    def test_get_event_history_unsupported(self):
        """reinforcement_engine 无 get_event_history 方法时返回空列表"""
        # 移除 get_event_history 属性
        del self.mock_reinforcement.get_event_history

        results = self.engine.get_event_history()

        assert results == []

    def test_get_archived_memories(self):
        """委托给 archiver.list_archived"""
        mock_records = [Mock()]
        self.mock_archiver.list_archived.return_value = mock_records

        results = self.engine.get_archived_memories(limit=50, vitality_threshold=10.0)

        self.mock_archiver.list_archived.assert_called_once_with(50, 10.0)
        assert results == mock_records

    def test_get_stats(self):
        """聚合各组件统计信息"""
        self.mock_gc.get_stats.return_value = {"collected": 5}
        self.mock_reinforcement.get_stats = Mock(return_value={"events": 10})
        # 确保 archiver 没有 _index 属性，避免 Mock 自动创建
        self.mock_archiver.configure_mock(**{"_index": {}})

        stats = self.engine.get_stats()

        assert stats["garbage_collector"] == {"collected": 5}
        assert stats["reinforcement"] == {"events": 10}
        assert stats["archive"] == {"total_archived": 0}

    def test_get_stats_without_optional_methods(self):
        """组件缺少 get_stats 时不报错"""
        del self.mock_gc.get_stats
        del self.mock_reinforcement.get_stats
        # 使用 spec 限制 archiver，使其没有 _index
        self.engine.archiver = Mock(spec=["archive", "resurrect", "list_archived"])

        stats = self.engine.get_stats()

        assert "garbage_collector" in stats
