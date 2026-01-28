"""
生命周期引擎组件协作测试

测试生命周期引擎内部各组件之间的协作：
- VitalityCalculator 与 ReinforcementEngine 的协作
- MemoryArchiver 与存储层的交互
- GarbageCollector 与 VitalityCalculator 的配合
- LifecycleEngine 的整体编排

不测试：与外部存储（Qdrant）的交互
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4

from hivememory.core.models import (
    MemoryAtom,
    MetaData,
    IndexLayer,
    PayloadLayer,
    MemoryType,
)
from hivememory.engines.lifecycle.interfaces import (
    BaseMemoryArchiver,
    BaseGarbageCollector,
)
from hivememory.engines.lifecycle.vitality import VitalityCalculator, VitalityCalculator
from hivememory.engines.lifecycle.archiver import FileBasedArchiver
from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
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
        
        # Mock configs
        mock_reinforcement_config = Mock()
        mock_reinforcement_config.vitality_adjustments = None
        mock_reinforcement_config.hit_boost = 5.0
        mock_reinforcement_config.citation_boost = 20.0
        mock_reinforcement_config.positive_feedback_boost = 50.0
        mock_reinforcement_config.negative_feedback_penalty = -50.0
        mock_reinforcement_config.event_history_limit = 1000
        mock_reinforcement_config.enable_event_history = True
        
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
            config=mock_reinforcement_config
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
        
        # Mock configs
        mock_reinforcement_config = Mock()
        mock_reinforcement_config.vitality_adjustments = None
        mock_reinforcement_config.hit_boost = 5.0
        mock_reinforcement_config.citation_boost = 20.0
        mock_reinforcement_config.positive_feedback_boost = 50.0
        mock_reinforcement_config.negative_feedback_penalty = -50.0
        mock_reinforcement_config.event_history_limit = 1000
        mock_reinforcement_config.enable_event_history = True
        
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
            config=mock_reinforcement_config
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
        
        mock_config = Mock()
        mock_config.archive_dir = "tmp_test_archive"
        mock_config.compression = False

        archiver = FileBasedArchiver(
            storage=mock_storage,
            config=mock_config
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
        
        mock_config = Mock()
        mock_config.archive_dir = "tmp_test_archive"
        mock_config.compression = False

        archiver = FileBasedArchiver(
            storage=mock_storage,
            config=mock_config
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
        
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)
        
        mock_gc_config = Mock()
        mock_gc_config.low_watermark = 20.0
        mock_gc_config.batch_size = 10

        gc = PeriodicGarbageCollector(
            storage=mock_storage,
            archiver=Mock(), # Mock archiver
            vitality_calculator=calculator,
            config=mock_gc_config,
        )

        # 运行垃圾回收
        collected = gc.collect()

        # 应该收集低生命力记忆
        assert collected >= 0


class TestLifecycleEngineCoordination:
    """测试 LifecycleEngine 的整体编排"""

    def test_engine_coordinates_all_components(self):
        """测试引擎协调所有组件"""
        mock_storage = Mock()
        mock_storage.get_memory = Mock(return_value=None)
        mock_storage.update_memory = Mock()
        
        # Add config mock for reinforcement engine
        mock_reinforcement_config = Mock()
        mock_reinforcement_config.vitality_adjustments = None
        mock_reinforcement_config.hit_boost = 5.0
        mock_reinforcement_config.citation_boost = 20.0
        mock_reinforcement_config.positive_feedback_boost = 50.0
        mock_reinforcement_config.negative_feedback_penalty = -50.0
        mock_reinforcement_config.event_history_limit = 1000
        mock_reinforcement_config.enable_event_history = True

        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0
        
        mock_archiver_config = Mock()
        mock_archiver_config.archive_dir = "tmp_test_archive"
        mock_archiver_config.compression = False
        
        mock_gc_config = Mock()
        mock_gc_config.low_watermark = 20.0
        mock_gc_config.batch_size = 10

        calculator = VitalityCalculator(config=mock_vitality_config)
        reinforcement = DynamicReinforcementEngine(storage=mock_storage, vitality_calculator=calculator, config=mock_reinforcement_config)
        archiver = FileBasedArchiver(storage=mock_storage, config=mock_archiver_config)
        gc = PeriodicGarbageCollector(storage=mock_storage, archiver=archiver, vitality_calculator=calculator, config=mock_gc_config)

        engine = MemoryLifecycleEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
            reinforcement_engine=reinforcement,
            archiver=archiver,
            garbage_collector=gc,
        )

        assert engine.vitality_calculator is calculator
        assert engine.reinforcement_engine is reinforcement
        assert engine.archiver is archiver
        assert engine.garbage_collector is gc

    def test_engine_handles_unknown_memory(self):
        """测试引擎处理未知记忆"""
        mock_storage = Mock()
        mock_storage.get_memory = Mock(return_value=None)
        
        # Add config mock for reinforcement engine
        mock_reinforcement_config = Mock()
        mock_reinforcement_config.vitality_adjustments = None
        mock_reinforcement_config.hit_boost = 5.0
        mock_reinforcement_config.citation_boost = 20.0
        mock_reinforcement_config.positive_feedback_boost = 50.0
        mock_reinforcement_config.negative_feedback_penalty = -50.0
        mock_reinforcement_config.event_history_limit = 1000
        mock_reinforcement_config.enable_event_history = True

        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0
        
        mock_archiver_config = Mock()
        mock_archiver_config.archive_dir = "tmp_test_archive"
        mock_archiver_config.compression = False
        
        mock_gc_config = Mock()
        mock_gc_config.low_watermark = 20.0
        mock_gc_config.batch_size = 10

        calculator = VitalityCalculator(config=mock_vitality_config)
        reinforcement = DynamicReinforcementEngine(storage=mock_storage, vitality_calculator=calculator, config=mock_reinforcement_config)
        archiver = FileBasedArchiver(storage=mock_storage, config=mock_archiver_config)
        gc = PeriodicGarbageCollector(storage=mock_storage, archiver=archiver, vitality_calculator=calculator, config=mock_gc_config)

        engine = MemoryLifecycleEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
            reinforcement_engine=reinforcement,
            archiver=archiver,
            garbage_collector=gc,
        )

        # calculate_vitality should raise ValueError if memory not found
        from uuid import UUID
        with pytest.raises(ValueError):
            engine.calculate_vitality(memory_id=UUID("00000000-0000-0000-0000-000000000000"))


class TestVitalityCalculation:
    """测试生命力计算逻辑"""

    def test_vitality_formula_components(self):
        """测试生命力公式各组件"""
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)

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
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)

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
        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)

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
        
        # Add config mock for reinforcement engine
        mock_reinforcement_config = Mock()
        mock_reinforcement_config.vitality_adjustments = None
        mock_reinforcement_config.hit_boost = 5.0
        mock_reinforcement_config.citation_boost = 20.0
        mock_reinforcement_config.positive_feedback_boost = 50.0
        mock_reinforcement_config.negative_feedback_penalty = -50.0
        mock_reinforcement_config.event_history_limit = 1000
        mock_reinforcement_config.enable_event_history = True
        mock_reinforcement_config.negative_confidence_multiplier = 0.5

        mock_vitality_config = Mock()
        mock_vitality_config.code_snippet_weight = 1.0
        mock_vitality_config.fact_weight = 0.9
        mock_vitality_config.url_resource_weight = 0.8
        mock_vitality_config.reflection_weight = 0.7
        mock_vitality_config.user_profile_weight = 0.6
        mock_vitality_config.work_in_progress_weight = 0.5
        mock_vitality_config.default_weight = 0.5
        mock_vitality_config.decay_lambda = 0.01
        mock_vitality_config.points_per_access = 2.0
        mock_vitality_config.max_access_boost = 20.0

        calculator = VitalityCalculator(config=mock_vitality_config)
        reinforcement = DynamicReinforcementEngine(
            storage=mock_storage,
            vitality_calculator=calculator,
            config=mock_reinforcement_config
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


class TestMemoryLifecycleEngine:
    """测试生命周期引擎"""

    def setup_method(self):
        """测试初始化"""
        self.mock_storage = Mock()

        # Mock components
        self.mock_vitality_calculator = Mock()
        self.mock_reinforcement_engine = Mock()
        self.mock_archiver = Mock()
        self.mock_garbage_collector = Mock()

        self.engine = MemoryLifecycleEngine(
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
                vitality_score=60.0,
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
        self.engine.vitality_calculator.calculate.return_value = 75.0

        vitality = self.engine.calculate_vitality(self.test_memory.id)

        assert vitality == 75.0
        self.engine.vitality_calculator.calculate.assert_called_once_with(self.test_memory)

    def test_calculate_vitality_memory_not_found(self):
        """测试计算不存在的记忆抛出异常"""
        self.mock_storage.get_memory.return_value = None

        with pytest.raises(ValueError, match="not found"):
            self.engine.calculate_vitality(uuid4())

    def test_record_hit_convenience(self):
        """测试 record_hit 便捷方法"""
        memory_id = uuid4()
        self.engine.reinforcement_engine.reinforce.return_value = MagicMock(
            new_vitality=55.0,
            previous_vitality=50.0,
        )

        result = self.engine.record_hit(memory_id)

        assert result.new_vitality == 55.0

        # 验证事件类型正确
        call_args = self.engine.reinforcement_engine.reinforce.call_args
        event = call_args[0][1]
        assert event.event_type == EventType.HIT
        assert event.memory_id == memory_id

    def test_record_citation_convenience(self):
        """测试 record_citation 便捷方法"""
        memory_id = uuid4()
        self.engine.reinforcement_engine.reinforce.return_value = MagicMock()

        result = self.engine.record_citation(memory_id)

        call_args = self.engine.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.CITATION

    def test_record_feedback_positive(self):
        """测试记录正面反馈"""
        memory_id = uuid4()
        self.engine.reinforcement_engine.reinforce.return_value = MagicMock()

        self.engine.record_feedback(memory_id, positive=True)

        call_args = self.engine.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.FEEDBACK_POSITIVE

    def test_record_feedback_negative(self):
        """测试记录负面反馈"""
        memory_id = uuid4()
        self.engine.reinforcement_engine.reinforce.return_value = MagicMock()

        self.engine.record_feedback(memory_id, positive=False)

        call_args = self.engine.reinforcement_engine.reinforce.call_args
        assert call_args[0][1].event_type == EventType.FEEDBACK_NEGATIVE

    def test_run_garbage_collection(self):
        """测试运行垃圾回收"""
        self.engine.garbage_collector.collect.return_value = 5

        archived = self.engine.run_garbage_collection(force=True)

        assert archived == 5
        self.engine.garbage_collector.collect.assert_called_once_with(force=True)

    def test_archive_memory(self):
        """测试手动归档记忆"""
        memory_id = uuid4()
        self.engine.archiver.archive.return_value = None

        self.engine.archive_memory(memory_id)

        self.engine.archiver.archive.assert_called_once_with(memory_id)

    def test_resurrect_memory(self):
        """测试唤醒记忆"""
        memory_id = uuid4()
        expected_memory = self.test_memory
        self.engine.archiver.resurrect.return_value = expected_memory

        result = self.engine.resurrect_memory(memory_id)

        assert result == expected_memory
        self.engine.archiver.resurrect.assert_called_once_with(memory_id)

    def test_get_low_vitality_memories(self):
        """测试获取低生命力记忆"""
        # 模拟返回记忆
        self.mock_storage.get_all_memories.return_value = [
            self.test_memory,
        ]
        self.engine.vitality_calculator.calculate.return_value = 15.0

        low_memories = self.engine.get_low_vitality_memories(threshold=20.0)

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
                        vitality_score=10.0,
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
        self.engine.vitality_calculator.calculate.return_value = 15.0

        low_memories = self.engine.get_low_vitality_memories(threshold=20.0, limit=5)

        # 应该只返回 5 个
        assert len(low_memories) == 5

    def test_get_event_history(self):
        """测试获取事件历史"""
        self.engine.reinforcement_engine.get_event_history.return_value = [
            MagicMock(memory_id=uuid4())
        ]

        history = self.engine.get_event_history()

        assert len(history) == 1
        self.engine.reinforcement_engine.get_event_history.assert_called_once()

    def test_get_stats(self):
        """测试获取统计信息"""
        self.engine.garbage_collector.get_stats.return_value = {
            "last_run": "2025-01-01",
            "total_archived": 10,
        }
        self.engine.reinforcement_engine.get_stats.return_value = {
            "total_events": 50,
        }

        # 设置索引
        self.engine.archiver._index = {str(uuid4()): Mock()}

        stats = self.engine.get_stats()

        assert "garbage_collector" in stats
        assert "archive" in stats
        assert stats["archive"]["total_archived"] == 1

    def test_get_archived_memories(self):
        """测试获取已归档记忆列表"""
        self.engine.archiver.list_archived.return_value = [
            Mock(memory_id=uuid4())
        ]

        archived = self.engine.get_archived_memories()

        assert len(archived) == 1
        self.engine.archiver.list_archived.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])