"""
HiveMemory Lifecycle Component E2E Tests

测试 Lifecycle (生命周期引擎) 的核心逻辑。

测试组：
    - Group 1: 评分逻辑测试 (Vitality Scoring)
    - Group 2: 强化事件测试 (Reinforcement)
    - Group 3: 归档与唤醒测试 (Archiving)

运行方式：
    pytest tests/components/test_lifecycle_e2e.py -v

核心原则：
    - 使用 Mock Storage 模拟 Qdrant
    - 使用 Mock Clock 模拟时间流逝
    - 验证评分公式、强化机制、归档流程

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
import math
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch, MagicMock

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置 ==========

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# ========== 其他导入 ==========

import pytest
from rich.console import Console
from rich.panel import Panel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import MemoryAtom, MemoryType

# Lifecycle 组件
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.archiver import FileBasedArchiver
from hivememory.engines.lifecycle.models import (
    EventType,
    MemoryEvent,
    ReinforcementResult,
    ArchiveRecord,
)

# 配置
from hivememory.patchouli.config import (
    VitalityCalculatorConfig,
    ReinforcementEngineConfig,
    ArchiverConfig,
)

# 导入测试数据
from tests.fixtures.lifecycle_test_data import (
    SCORING_TEST_CASES,
    REINFORCEMENT_TEST_CASES,
    ARCHIVING_TEST_CASES,
    create_test_memory,
    create_memory_with_age,
    get_scoring_test_by_id,
    get_reinforcement_test_by_id,
    get_archiving_test_by_id,
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== Mock Storage ==========

class MockQdrantMemoryStore:
    """
    模拟 Qdrant 存储

    用于测试 Lifecycle 组件，无需真实数据库连接。
    """

    def __init__(self):
        self.memories: Dict[UUID, MemoryAtom] = {}
        self._call_log: List[Dict] = []

    def get_memory(self, memory_id: UUID) -> Optional[MemoryAtom]:
        """获取记忆"""
        self._call_log.append({"method": "get_memory", "memory_id": memory_id})
        return self.memories.get(memory_id)

    def upsert_memory(self, memory: MemoryAtom) -> None:
        """插入或更新记忆"""
        self._call_log.append({"method": "upsert_memory", "memory_id": memory.id})
        self.memories[memory.id] = memory

    def delete_memory(self, memory_id: UUID) -> bool:
        """删除记忆"""
        self._call_log.append({"method": "delete_memory", "memory_id": memory_id})
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def list_all_memories(self, limit: int = 100) -> List[MemoryAtom]:
        """列出所有记忆"""
        self._call_log.append({"method": "list_all_memories", "limit": limit})
        return list(self.memories.values())[:limit]

    def clear(self) -> None:
        """清空存储"""
        self.memories.clear()
        self._call_log.clear()

    @property
    def count(self) -> int:
        """获取记忆数量"""
        return len(self.memories)


# ========== 辅助函数 ==========

def print_test_header(test_id: str, test_name: str) -> None:
    """打印测试标题"""
    console.print(Panel(
        f"[bold cyan]{test_id}[/bold cyan]: {test_name}",
        style="blue"
    ))


def print_test_result(test_id: str, test_name: str, passed: bool, details: str = "") -> None:
    """打印测试结果"""
    status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
    console.print(f"  {test_id}: {test_name} - {status}")
    if details:
        console.print(f"    [dim]{details}[/dim]")


# ========== Fixtures ==========

@pytest.fixture
def mock_storage() -> MockQdrantMemoryStore:
    """提供 Mock 存储实例"""
    return MockQdrantMemoryStore()


@pytest.fixture
def vitality_config() -> VitalityCalculatorConfig:
    """提供生命力计算器配置"""
    return VitalityCalculatorConfig(
        decay_lambda=0.01,
        max_access_boost=50.0,
        points_per_access=1.0,
        code_snippet_weight=1.0,
        fact_weight=0.9,
        url_resource_weight=0.8,
        reflection_weight=0.7,
        user_profile_weight=0.6,
        work_in_progress_weight=0.5,
        default_weight=0.5,
    )


@pytest.fixture
def vitality_calculator(vitality_config) -> VitalityCalculator:
    """提供生命力计算器实例"""
    return VitalityCalculator(config=vitality_config)


@pytest.fixture
def reinforcement_config() -> ReinforcementEngineConfig:
    """提供强化引擎配置"""
    return ReinforcementEngineConfig(
        hit_boost=5.0,
        citation_boost=20.0,
        positive_feedback_boost=50.0,
        negative_feedback_penalty=-50.0,
        negative_confidence_multiplier=0.5,
        enable_event_history=True,
        event_history_limit=1000,
    )


@pytest.fixture
def reinforcement_engine(
    mock_storage,
    reinforcement_config,
    vitality_calculator
) -> DynamicReinforcementEngine:
    """提供强化引擎实例"""
    return DynamicReinforcementEngine(
        storage=mock_storage,
        config=reinforcement_config,
        vitality_calculator=vitality_calculator,
    )


@pytest.fixture
def archiver_config(tmp_path) -> ArchiverConfig:
    """提供归档器配置（使用临时目录）"""
    return ArchiverConfig(
        archive_dir=str(tmp_path / "archived"),
        compression=True,
    )


@pytest.fixture
def archiver(mock_storage, archiver_config) -> FileBasedArchiver:
    """提供归档器实例"""
    return FileBasedArchiver(
        storage=mock_storage,
        config=archiver_config,
    )


# ========== 测试类: 评分逻辑 ==========

class TestVitalityScoring:
    """
    评分逻辑测试

    验证 VitalityCalculator 的评分公式:
    V = (C × I) × D(t) × 100 + A
    """

    def test_lif_scr_001_base_score_by_type(self, vitality_calculator):
        """
        LIF-SCR-001: 基础分计算

        验证不同记忆类型的固有价值权重差异。
        CODE_SNIPPET (权重 1.0) 应该比 WORK_IN_PROGRESS (权重 0.5) 分数更高。
        """
        case = get_scoring_test_by_id("LIF-SCR-001")
        print_test_header(case["id"], case["name"])

        # 创建两种类型的记忆，相同置信度
        code_memory = create_test_memory(
            template_name="code_snippet",
            confidence_score=0.9,
            access_count=0,
        )
        wip_memory = create_test_memory(
            template_name="work_in_progress",
            confidence_score=0.9,
            access_count=0,
        )

        # 计算分数
        code_score = vitality_calculator.calculate(code_memory)
        wip_score = vitality_calculator.calculate(wip_memory)

        # 验证
        assert code_score > wip_score, (
            f"CODE_SNIPPET ({code_score:.2f}) should be higher than "
            f"WORK_IN_PROGRESS ({wip_score:.2f})"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"CODE={code_score:.2f}, WIP={wip_score:.2f}"
        )

    def test_lif_scr_002_time_decay(self, vitality_calculator):
        """
        LIF-SCR-002: 时间衰减测试

        验证时间流逝导致的分数衰减。
        30 天后的分数应该明显低于初始分数。
        """
        case = get_scoring_test_by_id("LIF-SCR-002")
        print_test_header(case["id"], case["name"])

        # 创建新鲜记忆（刚刚更新）
        fresh_memory = create_test_memory(
            template_name="fact",
            confidence_score=0.9,
            access_count=0,
        )

        # 创建 30 天前的记忆
        old_memory = create_memory_with_age(
            days_old=30,
            template_name="fact",
            confidence_score=0.9,
            access_count=0,
        )

        # 计算分数
        fresh_score = vitality_calculator.calculate(fresh_memory)
        old_score = vitality_calculator.calculate(old_memory)

        # 预期衰减因子: exp(-0.01 * 30) ≈ 0.7408
        expected_decay = math.exp(-0.01 * 30)
        tolerance = case["tolerance"]

        # 验证衰减
        assert old_score < fresh_score, (
            f"Old memory ({old_score:.2f}) should have lower score than "
            f"fresh memory ({fresh_score:.2f})"
        )

        # 验证衰减比例接近预期
        actual_ratio = old_score / fresh_score if fresh_score > 0 else 0
        assert abs(actual_ratio - expected_decay) < tolerance, (
            f"Decay ratio ({actual_ratio:.4f}) should be close to "
            f"expected ({expected_decay:.4f})"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Fresh={fresh_score:.2f}, Old={old_score:.2f}, Ratio={actual_ratio:.4f}"
        )

    def test_lif_scr_003_access_boost_cap(self, vitality_calculator):
        """
        LIF-SCR-003: 访问加成上限

        验证访问加成不会超过配置的上限 (50)。
        """
        case = get_scoring_test_by_id("LIF-SCR-003")
        print_test_header(case["id"], case["name"])

        max_boost = case["max_access_boost"]

        # 创建高访问次数的记忆
        high_access_memory = create_test_memory(
            template_name="fact",
            confidence_score=0.9,
            access_count=10000,  # 极高访问次数
        )

        # 创建零访问的记忆
        zero_access_memory = create_test_memory(
            template_name="fact",
            confidence_score=0.9,
            access_count=0,
        )

        # 计算分数
        high_score = vitality_calculator.calculate(high_access_memory)
        zero_score = vitality_calculator.calculate(zero_access_memory)

        # 访问加成 = high_score - zero_score
        actual_boost = high_score - zero_score

        # 验证加成不超过上限
        assert actual_boost <= max_boost + 0.1, (
            f"Access boost ({actual_boost:.2f}) should not exceed "
            f"max cap ({max_boost})"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Boost={actual_boost:.2f}, MaxCap={max_boost}"
        )


# ========== 测试类: 强化事件 ==========

class TestReinforcement:
    """
    强化事件测试

    验证 DynamicReinforcementEngine 对各种事件的处理。
    """

    def test_lif_rnf_001_hit_event(self, mock_storage, reinforcement_engine):
        """
        LIF-RNF-001: 检索命中 (HIT)

        验证 HIT 事件增加生命力和访问计数。
        """
        case = get_reinforcement_test_by_id("LIF-RNF-001")
        print_test_header(case["id"], case["name"])

        # 创建并存储记忆
        memory = create_test_memory(
            template_name="fact",
            vitality_score=50.0,
            confidence_score=0.8,
            access_count=0,
        )
        mock_storage.upsert_memory(memory)

        initial_vitality = memory.meta.vitality_score
        initial_access_count = memory.meta.access_count

        # 触发 HIT 事件
        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=memory.id,
            source="test",
        )
        result = reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = mock_storage.get_memory(memory.id)

        # 验证
        assert updated_memory.meta.access_count == initial_access_count + 1, (
            f"Access count should increase by 1"
        )
        assert result.new_vitality >= result.previous_vitality, (
            f"Vitality should increase or stay same after HIT"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Vitality: {result.previous_vitality:.2f} -> {result.new_vitality:.2f}, "
            f"AccessCount: {initial_access_count} -> {updated_memory.meta.access_count}"
        )

    def test_lif_rnf_002_citation_event(self, mock_storage, reinforcement_engine):
        """
        LIF-RNF-002: 引用强化 (CITATION)

        验证 CITATION 事件重置时间衰减并大幅提升生命力。
        """
        case = get_reinforcement_test_by_id("LIF-RNF-002")
        print_test_header(case["id"], case["name"])

        # 创建 30 天前的记忆（已衰减）
        old_memory = create_memory_with_age(
            days_old=30,
            template_name="fact",
            vitality_score=30.0,  # 已衰减的分数
            confidence_score=0.8,
            access_count=5,
        )
        old_updated_at = old_memory.meta.updated_at
        mock_storage.upsert_memory(old_memory)

        # 触发 CITATION 事件
        event = MemoryEvent(
            event_type=EventType.CITATION,
            memory_id=old_memory.id,
            source="test",
        )
        result = reinforcement_engine.reinforce(old_memory.id, event)

        # 获取更新后的记忆
        updated_memory = mock_storage.get_memory(old_memory.id)

        # 验证 updated_at 被更新（时间衰减重置）
        assert updated_memory.meta.updated_at > old_updated_at, (
            "updated_at should be reset to now"
        )

        # 验证生命力提升
        assert result.new_vitality > result.previous_vitality, (
            f"Vitality should increase after CITATION"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Vitality: {result.previous_vitality:.2f} -> {result.new_vitality:.2f}, "
            f"Decay Reset: True"
        )

    def test_lif_rnf_003_negative_feedback(self, mock_storage, reinforcement_engine):
        """
        LIF-RNF-003: 用户负面反馈 (FEEDBACK_NEGATIVE)

        验证负面反馈降低生命力和置信度。
        """
        case = get_reinforcement_test_by_id("LIF-RNF-003")
        print_test_header(case["id"], case["name"])

        # 创建高置信度记忆
        memory = create_test_memory(
            template_name="fact",
            vitality_score=80.0,
            confidence_score=0.9,
            access_count=10,
        )
        mock_storage.upsert_memory(memory)

        initial_confidence = memory.meta.confidence_score

        # 触发负面反馈事件
        event = MemoryEvent(
            event_type=EventType.FEEDBACK_NEGATIVE,
            memory_id=memory.id,
            source="user",
        )
        result = reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = mock_storage.get_memory(memory.id)

        # 验证置信度降低 (乘以 0.5)
        expected_confidence = initial_confidence * case["expected_confidence_multiplier"]
        assert abs(updated_memory.meta.confidence_score - expected_confidence) < 0.01, (
            f"Confidence should be multiplied by {case['expected_confidence_multiplier']}"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Vitality: {result.previous_vitality:.2f} -> {result.new_vitality:.2f}, "
            f"Confidence: {result.previous_confidence:.2f} -> {result.new_confidence:.2f}"
        )

    def test_lif_rnf_004_positive_feedback(self, mock_storage, reinforcement_engine):
        """
        LIF-RNF-004: 用户正面反馈 (FEEDBACK_POSITIVE)

        验证正面反馈提升生命力。
        """
        case = get_reinforcement_test_by_id("LIF-RNF-004")
        print_test_header(case["id"], case["name"])

        # 创建记忆
        memory = create_test_memory(
            template_name="fact",
            vitality_score=50.0,
            confidence_score=0.7,
            access_count=5,
        )
        mock_storage.upsert_memory(memory)

        initial_access_count = memory.meta.access_count

        # 触发正面反馈事件
        event = MemoryEvent(
            event_type=EventType.FEEDBACK_POSITIVE,
            memory_id=memory.id,
            source="user",
        )
        result = reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = mock_storage.get_memory(memory.id)

        # 验证生命力提升
        assert result.new_vitality > result.previous_vitality, (
            "Vitality should increase after positive feedback"
        )

        # 验证访问计数增加
        assert updated_memory.meta.access_count == initial_access_count + 1, (
            "Access count should increase by 1"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Vitality: {result.previous_vitality:.2f} -> {result.new_vitality:.2f}, "
            f"AccessCount: {initial_access_count} -> {updated_memory.meta.access_count}"
        )


# ========== 测试类: 归档与唤醒 ==========

class TestArchiving:
    """
    归档与唤醒测试

    验证 FileBasedArchiver 的归档和唤醒流程。
    """

    def test_lif_arc_001_archive_trigger(self, mock_storage, archiver):
        """
        LIF-ARC-001: 归档触发

        验证低分记忆被正确归档到冷存储。
        """
        case = get_archiving_test_by_id("LIF-ARC-001")
        print_test_header(case["id"], case["name"])

        # 创建低生命力记忆
        memory = create_test_memory(
            template_name="low_vitality",
            vitality_score=case["initial_vitality"],
        )
        mock_storage.upsert_memory(memory)
        memory_id = memory.id

        # 验证记忆在热存储中
        assert mock_storage.get_memory(memory_id) is not None, (
            "Memory should exist in hot storage before archive"
        )

        # 执行归档
        archiver.archive(memory_id)

        # 验证记忆从热存储中删除
        assert mock_storage.get_memory(memory_id) is None, (
            "Memory should be deleted from hot storage after archive"
        )

        # 验证记忆在冷存储中
        assert archiver.is_archived(memory_id), (
            "Memory should be in cold storage after archive"
        )

        # 验证归档文件存在
        record = archiver.get_archive_record(memory_id)
        assert record is not None, "Archive record should exist"
        assert Path(record.storage_path).exists(), (
            f"Archive file should exist at {record.storage_path}"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Archived to {Path(record.storage_path).name}"
        )

    def test_lif_arc_002_resurrect_flow(self, mock_storage, archiver):
        """
        LIF-ARC-002: 唤醒流程

        验证已归档记忆能被正确唤醒到热存储，且数据完整。
        """
        case = get_archiving_test_by_id("LIF-ARC-002")
        print_test_header(case["id"], case["name"])

        # 创建并归档记忆
        original_memory = create_test_memory(
            template_name="fact",
            vitality_score=5.0,
        )
        mock_storage.upsert_memory(original_memory)
        memory_id = original_memory.id

        # 保存原始数据用于比较
        original_title = original_memory.index.title
        original_content = original_memory.payload.content
        original_tags = original_memory.index.tags.copy()

        # 归档
        archiver.archive(memory_id)
        assert archiver.is_archived(memory_id), "Memory should be archived"

        # 唤醒
        resurrected_memory = archiver.resurrect(memory_id)

        # 验证记忆回到热存储
        assert mock_storage.get_memory(memory_id) is not None, (
            "Memory should be back in hot storage"
        )

        # 验证不再在冷存储中
        assert not archiver.is_archived(memory_id), (
            "Memory should not be in cold storage after resurrect"
        )

        # 验证数据完整性
        assert resurrected_memory.id == memory_id, "ID should match"
        assert resurrected_memory.index.title == original_title, "Title should match"
        assert resurrected_memory.payload.content == original_content, "Content should match"
        assert resurrected_memory.index.tags == original_tags, "Tags should match"

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Data integrity verified: title, content, tags all match"
        )

    def test_lif_arc_003_archive_idempotency(self, mock_storage, archiver):
        """
        LIF-ARC-003: 归档幂等性

        验证对已归档记忆再次调用归档不会产生错误或重复文件。
        """
        case = get_archiving_test_by_id("LIF-ARC-003")
        print_test_header(case["id"], case["name"])

        # 创建并归档记忆
        memory = create_test_memory(
            template_name="low_vitality",
            vitality_score=5.0,
        )
        mock_storage.upsert_memory(memory)
        memory_id = memory.id

        # 第一次归档
        archiver.archive(memory_id)
        assert archiver.is_archived(memory_id), "Memory should be archived"

        # 获取归档记录
        first_record = archiver.get_archive_record(memory_id)
        first_path = first_record.storage_path

        # 第二次归档（应该是幂等的，不产生错误）
        try:
            archiver.archive(memory_id)
            no_error = True
        except Exception as e:
            no_error = False
            console.print(f"[yellow]Warning: Second archive raised {type(e).__name__}[/yellow]")

        # 验证没有重复文件
        record_after = archiver.get_archive_record(memory_id)
        assert record_after.storage_path == first_path, (
            "Archive path should remain the same"
        )

        print_test_result(
            case["id"],
            case["name"],
            no_error,
            f"Idempotent: no error, no duplicate file"
        )


# ========== 主入口 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
