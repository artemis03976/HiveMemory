"""
HiveMemory - 生命周期管理器集成测试

测试内容:
- 完整生命周期工作流
- 组件协调
- 统一接口
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.lifecycle.orchestrator import MemoryLifecycleManager, create_default_lifecycle_manager
from hivememory.lifecycle.types import MemoryEvent, EventType


class TestMemoryLifecycleManager:
    """测试生命周期管理器"""

    def setup_method(self):
        """测试初始化"""
        self.mock_storage = Mock()

        # Mock components
        self.mock_vitality_calculator = Mock()
        self.mock_reinforcement_engine = Mock()
        self.mock_archiver = Mock()
        self.mock_garbage_collector = Mock()

        self.manager = MemoryLifecycleManager(
            storage=self.mock_storage,
            vitality_calculator=self.mock_vitality_calculator,
            reinforcement_engine=self.mock_reinforcement_engine,
            archiver=self.mock_archiver,
            garbage_collector=self.mock_garbage_collector
        )

        # 创建测试记忆
        self.test_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=0.6,
            ),
            index=IndexLayer(
                title="Test",
                summary="Test summary with enough length",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

    def teardown_method(self):
        """清理"""
        pass

    def test_calculate_vitality(self):
        """测试计算生命力"""
        self.mock_storage.get_memory.return_value = self.test_memory
        self.manager.vitality_calculator.calculate.return_value = 75.0

        vitality = self.manager.calculate_vitality(self.test_memory.id)

        assert vitality == 75.0
        self.manager.vitality_calculator.calculate.assert_called_once_with(self.test_memory)

    def test_calculate_vitality_memory_not_found(self):
        """测试计算不存在的记忆抛出异常"""
        self.mock_storage.get_memory.return_value = None

        with pytest.raises(ValueError, match="not found"):
            self.manager.calculate_vitality(uuid4())

    def test_record_hit_convenience(self):
        """测试 record_hit 便捷方法"""
        memory_id = uuid4()
        self.manager.reinforcement_engine.reinforce.return_value = MagicMock(
            new_vitality=55.0,
            previous_vitality=50.0,
        )

        result = self.manager.record_hit(memory_id)

        assert result.new_vitality == 55.0

        # 验证事件类型正确
        call_args = self.manager.reinforcement_engine.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.HIT
        assert event.memory_id == memory_id

    def test_record_citation_convenience(self):
        """测试 record_citation 便捷方法"""
        memory_id = uuid4()
        self.manager.reinforcement_engine.reinforce.return_value = MagicMock()

        result = self.manager.record_citation(memory_id)

        call_args = self.manager.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.CITATION

    def test_record_feedback_positive(self):
        """测试记录正面反馈"""
        memory_id = uuid4()
        self.manager.reinforcement_engine.reinforce.return_value = MagicMock()

        self.manager.record_feedback(memory_id, positive=True)

        call_args = self.manager.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.FEEDBACK_POSITIVE

    def test_record_feedback_negative(self):
        """测试记录负面反馈"""
        memory_id = uuid4()
        self.manager.reinforcement_engine.reinforce.return_value = MagicMock()

        self.manager.record_feedback(memory_id, positive=False)

        call_args = self.manager.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.FEEDBACK_NEGATIVE

    def test_run_garbage_collection(self):
        """测试运行垃圾回收"""
        self.manager.garbage_collector.collect.return_value = 5

        archived = self.manager.run_garbage_collection(force=True)

        assert archived == 5
        self.manager.garbage_collector.collect.assert_called_once_with(force=True)

    def test_archive_memory(self):
        """测试手动归档记忆"""
        memory_id = uuid4()
        self.manager.archiver.archive.return_value = None

        self.manager.archive_memory(memory_id)

        self.manager.archiver.archive.assert_called_once_with(memory_id)

    def test_resurrect_memory(self):
        """测试唤醒记忆"""
        memory_id = uuid4()
        expected_memory = self.test_memory
        self.manager.archiver.resurrect.return_value = expected_memory

        result = self.manager.resurrect_memory(memory_id)

        assert result == expected_memory
        self.manager.archiver.resurrect.assert_called_once_with(memory_id)

    def test_get_low_vitality_memories(self):
        """测试获取低生命力记忆"""
        # 模拟返回记忆
        self.mock_storage.get_all_memories.return_value = [
            self.test_memory,
        ]
        self.manager.vitality_calculator.calculate.return_value = 15.0

        low_memories = self.manager.get_low_vitality_memories(threshold=20.0)

        assert len(low_memories) == 1
        assert low_memories[0][0] == self.test_memory.id
        assert low_memories[0][1] == 15.0

    def test_get_low_vitality_memories_respects_limit(self):
        """测试获取低生命力记忆时遵守限制"""
        # 创建多个记忆
        memories = []
        for i in range(10):
            memories.append(
                MemoryAtom(
                    id=uuid4(),
                    meta=MetaData(
                        source_agent_id="a",
                        user_id="u",
                        vitality_score=0.1,
                        confidence_score=0.8,
                    ),
                    index=IndexLayer(
                        title=f"M{i}",
                        summary="summary limit test",
                        tags=[],
                        memory_type=MemoryType.FACT,
                    ),
                    payload=PayloadLayer(content="c"),
                )
            )

        self.mock_storage.get_all_memories.return_value = memories
        self.manager.vitality_calculator.calculate.return_value = 15.0

        low_memories = self.manager.get_low_vitality_memories(threshold=20.0, limit=5)

        # 应该只返回 5 个
        assert len(low_memories) == 5

    def test_get_event_history(self):
        """测试获取事件历史"""
        self.manager.reinforcement_engine.get_event_history.return_value = [
            MagicMock(memory_id=uuid4())
        ]

        history = self.manager.get_event_history()

        assert len(history) == 1
        self.manager.reinforcement_engine.get_event_history.assert_called_once()

    def test_get_stats(self):
        """测试获取统计信息"""
        self.manager.garbage_collector.get_stats.return_value = {
            "last_run": "2025-01-01",
            "total_archived": 10,
        }
        self.manager.reinforcement_engine.get_stats.return_value = {
            "total_events": 50,
        }

        # 设置索引
        self.manager.archiver._index = {str(uuid4()): Mock()}

        stats = self.manager.get_stats()

        assert "garbage_collector" in stats
        assert "archive" in stats
        assert stats["archive"]["total_archived"] == 1

    def test_get_archived_memories(self):
        """测试获取已归档记忆列表"""
        self.manager.archiver.list_archived.return_value = [
            Mock(memory_id=uuid4())
        ]

        archived = self.manager.get_archived_memories()

        assert len(archived) == 1
        self.manager.archiver.list_archived.assert_called_once()


class TestCreateDefaultLifecycleManager:
    """测试默认生命周期管理器工厂函数"""

    @patch('hivememory.lifecycle.garbage_collector.create_default_garbage_collector')
    @patch('hivememory.lifecycle.archiver.create_default_archiver')
    @patch('hivememory.lifecycle.reinforcement.create_default_reinforcement_engine')
    @patch('hivememory.lifecycle.vitality.create_default_vitality_calculator')
    def test_create_default_manager(
        self,
        mock_gc,
        mock_archiver,
        mock_reinforcement,
        mock_vitality
    ):
        """测试创建默认管理器"""
        mock_storage = Mock()
        mock_vitality.return_value = Mock()
        mock_reinforcement.return_value = Mock()
        mock_archiver.return_value = Mock()
        mock_gc.return_value = Mock()

        manager = create_default_lifecycle_manager(mock_storage)

        assert isinstance(manager, MemoryLifecycleManager)
        assert manager.storage == mock_storage
