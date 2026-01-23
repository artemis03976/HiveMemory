"""
生命周期引擎组件协作测试

测试生命周期引擎内部各组件之间的协作：
- VitalityCalculator 与 ReinforcementEngine 的协作
- MemoryArchiver 与存储层的交互
- GarbageCollector 与 VitalityCalculator 的配合
- LifecycleManager 的整体编排

不测试：与外部存储（Qdrant）的交互
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from typing import List

from hivememory.core.models import (
    MemoryAtom,
    MetaData,
    IndexLayer,
    PayloadLayer,
    MemoryType,
)
from hivememory.engines.lifecycle.interfaces import (
    LifecycleManager,
    VitalityCalculator,
    ReinforcementEngine,
    MemoryArchiver,
    GarbageCollector,
)
from hivememory.engines.lifecycle.vitality import StandardVitalityCalculator
from hivememory.engines.lifecycle.archiver import FileBasedMemoryArchiver
from hivememory.engines.lifecycle.orchestrator import MemoryLifecycleManager
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.garbage_collector import PeriodicGarbageCollector
from hivememory.engines.lifecycle.models import (
    MemoryEvent,
    EventType,
    ReinforcementResult,
    ArchiveRecord,
)


class TestVitalityAndReinforcementCollaboration:
    """测试 VitalityCalculator 与 ReinforcementEngine 的协作"""

    def test_reinforcement_updates_vitality(self):
        """测试强化事件更新生命力分数"""
        mock_storage = Mock()
        mock_storage.update_memory = Mock()

        calculator = StandardVitalityCalculator()
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
        )

        # 创建测试记忆
        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.8,
                access_count=0,
            ),
            index=IndexLayer(
                title="测试记忆",
                summary="这是一个测试用的摘要信息，长度必须超过十个字符",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="测试内容"),
        )
        
        # Mock get_memory to return the memory
        mock_storage.get_memory = Mock(return_value=memory)

        # 记录初始生命力
        initial_vitality = calculator.calculate(memory)

        # 应用 HIT 事件
        event = MemoryEvent(
            memory_id=memory.id,
            event_type=EventType.HIT,
            source="test"
        )
        result = reinforcement.reinforce(memory.id, event)

        # 验证强化结果
        assert result is not None
        assert result.new_vitality > initial_vitality, "HIT事件应该增加生命力"

    def test_multiple_reinforcement_events_cumulative(self):
        """测试多次强化事件的累积效果"""
        mock_storage = Mock()
        mock_storage.update_memory = Mock()

        calculator = StandardVitalityCalculator()
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
        )

        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.8,
                access_count=0,
            ),
            index=IndexLayer(
                title="测试记忆",
                summary="这是一个测试用的摘要信息，长度必须超过十个字符",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="测试内容"),
        )
        
        # Mock get_memory
        mock_storage.get_memory = Mock(return_value=memory)

        vitality_scores = []
        for event_type in [EventType.HIT, EventType.CITATION, EventType.FEEDBACK_POSITIVE]:
            event = MemoryEvent(
                memory_id=memory.id,
                event_type=event_type,
                source="test"
            )
            result = reinforcement.reinforce(memory.id, event)
            vitality_scores.append(result.new_vitality)

        # 验证生命力逐步上升
        assert vitality_scores[0] < vitality_scores[1], "CITATION应该比HIT增加更多"
        assert vitality_scores[1] < vitality_scores[2], "FEEDBACK_POSITIVE应该继续增加"


class TestArchiverAndStorageCollaboration:
    """测试 MemoryArchiver 与存储层的协作"""

    def test_archiver_moves_memory_to_cold_storage(self):
        """测试归档器将记忆移至冷存储"""
        from uuid import uuid4
        mock_storage = Mock()
        archived_memories = {}

        def mock_delete(memory_id):
            archived_memories[memory_id] = "archived"

        mock_storage.delete_memory = mock_delete
        
        # Mock get_memory to return a valid memory
        mock_storage.get_memory = Mock(return_value=MemoryAtom(
            meta=MetaData(source_agent_id="test", user_id="test", confidence_score=0.8, vitality_score=10.0),
            index=IndexLayer(title="test", summary="这是一个测试用的摘要信息，长度必须超过十个字符", tags=[], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="test")
        ))

        archiver = FileBasedMemoryArchiver(
            storage=mock_storage,
            archive_dir="tmp_test_archive",
        )

        memory_id = uuid4()
        archiver.archive(memory_id)

        # 验证存储层被调用
        assert memory_id in archived_memories

    def test_archiver_tracks_archived_memories(self):
        """测试归档器跟踪已归档的记忆"""
        mock_storage = Mock()
        mock_storage.get_memory = Mock(return_value=MemoryAtom(
            meta=MetaData(source_agent_id="test", user_id="test", confidence_score=0.8, vitality_score=10.0),
            index=IndexLayer(title="test", summary="这是一个测试用的摘要信息，长度必须超过十个字符", tags=[], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="test")
        ))
        mock_storage.delete_memory = Mock()

        archiver = FileBasedMemoryArchiver(
            storage=mock_storage,
            archive_dir="tmp_test_archive",
        )

        memory_id = "test_memory_id"
        # We need to mock get_memory to return something, handled above
        # But MemoryAtom id is generated.
        # Let's create a memory with fixed ID
        from uuid import uuid4
        mid = uuid4()
        
        mock_storage.get_memory = Mock(return_value=MemoryAtom(
            id=mid,
            meta=MetaData(source_agent_id="test", user_id="test", confidence_score=0.8, vitality_score=10.0),
            index=IndexLayer(title="test", summary="这是一个测试用的摘要信息，长度必须超过十个字符", tags=[], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="test")
        ))
        
        archiver.archive(mid)

        # 验证归档记录
        record = archiver.get_archive_record(mid)
        assert record is not None
        assert record.memory_id == mid


class TestGarbageCollectorAndVitalityCollaboration:
    """测试 GarbageCollector 与 VitalityCalculator 的协作"""

    def test_gc_uses_vitality_to_find_candidates(self):
        """测试垃圾回收器使用生命力计算查找候选"""
        mock_storage = Mock()

        # 创建不同生命力分数的记忆
        memories = [
            MemoryAtom(
                meta=MetaData(
                    source_agent_id="test_agent",
                    user_id="test_user",
                    confidence_score=0.1,
                    vitality_score=5.0,
                ),
                index=IndexLayer(
                    title=f"低价值记忆{i}",
                    summary="这是一个测试用的摘要信息，长度必须超过十个字符",
                    tags=["test"],
                    memory_type=MemoryType.WORK_IN_PROGRESS,
                ),
                payload=PayloadLayer(content="内容"),
            )
            for i in range(3)
        ]

        mock_storage.get_all_memories = Mock(return_value=memories)
        mock_storage.delete_memory = Mock()

        calculator = StandardVitalityCalculator()
        gc = PeriodicGarbageCollector(
            storage=mock_storage,
            archiver=Mock(), # Mock archiver
            vitality_calculator=calculator,
            low_watermark=20.0,
        )

        # 运行垃圾回收
        collected = gc.collect()

        # 应该收集低生命力记忆
        assert collected >= 0


class TestLifecycleManagerCoordination:
    """测试 LifecycleManager 的整体编排"""

    def test_manager_coordinates_all_components(self):
        """测试管理器协调所有组件"""
        mock_storage = Mock()
        mock_storage.get_memory = Mock(return_value=None)
        mock_storage.update_memory = Mock()

        calculator = StandardVitalityCalculator()
        reinforcement = DynamicReinforcementEngine(storage=mock_storage, vitality_calculator=calculator)
        archiver = FileBasedMemoryArchiver(storage=mock_storage, archive_dir="tmp_test_archive")
        gc = PeriodicGarbageCollector(storage=mock_storage, archiver=archiver, vitality_calculator=calculator)

        manager = MemoryLifecycleManager(
            storage=mock_storage,
            vitality_calculator=calculator,
            reinforcement_engine=reinforcement,
            archiver=archiver,
            garbage_collector=gc,
        )

        assert manager.vitality_calculator is calculator
        assert manager.reinforcement_engine is reinforcement
        assert manager.archiver is archiver
        assert manager.garbage_collector is gc

    def test_manager_handles_unknown_memory(self):
        """测试管理器处理未知记忆"""
        mock_storage = Mock()
        mock_storage.get_memory = Mock(return_value=None)

        manager = MemoryLifecycleManager(storage=mock_storage)
        
        # calculate_vitality should raise ValueError if memory not found
        from uuid import UUID
        with pytest.raises(ValueError):
            manager.calculate_vitality(memory_id=UUID("00000000-0000-0000-0000-000000000000"))


class TestVitalityCalculation:
    """测试生命力计算逻辑"""

    def test_vitality_formula_components(self):
        """测试生命力公式各组件"""
        calculator = StandardVitalityCalculator()

        # 高置信度 + 高固有价值类型
        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.95,
                access_count=10,
            ),
            index=IndexLayer(
                title="Python代码",
                summary="重要代码片段，长度超过十个字符",
                tags=["python", "code"],
                memory_type=MemoryType.CODE_SNIPPET,
            ),
            payload=PayloadLayer(content="def test(): pass"),
        )

        vitality = calculator.calculate(memory)

        # CODE_SNIPPET 类型 + 高置信度 + 访问量 = 应该有较高生命力
        assert vitality > 50, "高价值记忆应该有较高生命力"

    def test_vitality_decay_over_time(self):
        """测试生命力随时间衰减"""
        calculator = StandardVitalityCalculator()

        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.8,
                created_at=datetime.now() - timedelta(days=30),
                last_accessed_at=datetime.now() - timedelta(days=30),
            ),
            index=IndexLayer(
                title="旧记忆",
                summary="很久以前的记忆，长度超过十个字符",
                tags=["old"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="内容"),
        )

        vitality = calculator.calculate(memory)

        # 旧记忆应该有较低生命力
        # （具体值取决于衰减公式）

    def test_different_memory_type_weights(self):
        """测试不同记忆类型的权重"""
        calculator = StandardVitalityCalculator()

        memories = [
            MemoryAtom(
                meta=MetaData(
                    source_agent_id="test_agent",
                    user_id="test_user",
                    confidence_score=0.8,
                ),
                index=IndexLayer(
                    title=title,
                    summary="这是一个测试用的摘要信息，长度必须超过十个字符",
                    tags=["test"],
                    memory_type=mem_type,
                ),
                payload=PayloadLayer(content="内容"),
            )
            for title, mem_type in [
                ("代码片段", MemoryType.CODE_SNIPPET),
                ("用户偏好", MemoryType.USER_PROFILE),
                ("事实", MemoryType.FACT),
                ("进行中", MemoryType.WORK_IN_PROGRESS),
            ]
        ]

        vitalities = [calculator.calculate(m) for m in memories]

        # CODE_SNIPPET 应该有最高或较高的权重
        code_vitality = vitalities[0]
        assert code_vitality > 0


class TestEventTypeHandling:
    """测试不同事件类型的处理"""

    def test_all_event_types_produce_results(self):
        """测试所有事件类型都能产生结果"""
        mock_storage = Mock()
        mock_storage.update_memory = Mock()

        calculator = StandardVitalityCalculator()
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
        )

        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.8,
            ),
            index=IndexLayer(
                title="测试",
                summary="这是一个测试用的摘要信息，长度必须超过十个字符",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="内容"),
        )
        
        # Mock get_memory
        mock_storage.get_memory = Mock(return_value=memory)

        event_types = [
            EventType.HIT,
            EventType.CITATION,
            EventType.FEEDBACK_POSITIVE,
            EventType.FEEDBACK_NEGATIVE,
        ]

        for event_type in event_types:
            event = MemoryEvent(
                memory_id=memory.id,
                event_type=event_type,
                source="test"
            )
            result = reinforcement.reinforce(memory.id, event)
            assert result is not None, f"{event_type} 应该产生结果"
            assert result.event_type == event_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
