"""
HiveMemory Lifecycle 组件单元测试。

测试 Lifecycle (生命周期引擎) 的核心逻辑。

测试组：
    - Group 1: 评分逻辑测试 (Vitality Scoring)
    - Group 2: 强化事件测试 (Reinforcement)
    - Group 3: 归档与唤醒测试 (Archiving)

核心原则：
    - 使用 Mock Storage 模拟 Qdrant
    - 使用 Mock Clock 模拟时间流逝
    - 验证评分公式、强化机制、归档流程
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

# 核心模型
from hivememory.core.models import MemoryAtom, MemoryType

# Lifecycle 组件
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.models import (
    EventType,
    MemoryEvent,
    ReinforcementResult,
    ArchiveRecord,
)
from hivememory.patchouli.memory_library import (
    LongTermMemoryStore,
    MemoryLibrary,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter

# 配置
from hivememory.system.config import (
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


class _MockMidTermAdapter:
    def __init__(self, storage):
        self._storage = storage

    async def upsert(self, memory: MemoryAtom) -> None:
        await self._storage.upsert_memory(memory)

    async def get(self, memory_id: UUID) -> Optional[MemoryAtom]:
        return await self._storage.get_memory(memory_id)

    async def get_by_alias(self, alias: str, user_id: Optional[str] = None) -> Optional[MemoryAtom]:
        return None

    async def update_access_info(self, memory_id: UUID) -> None:
        return None

    async def delete(self, memory_id: UUID) -> bool:
        return await self._storage.delete_memory(memory_id)

    async def batch_delete(self, ids: List[UUID]) -> int:
        count = 0
        for memory_id in ids:
            if await self.delete(memory_id):
                count += 1
        return count

    async def search(self, query: str, top_k: int, filters=None, mode: str = "dense", score_threshold: float = 0.0):
        return []

    async def scroll(self, filters=None, limit: int = 100) -> List[MemoryAtom]:
        return self._storage.list_all_memories(limit=limit)

    async def count(self, filters=None) -> int:
        return self._storage.count


class _LegacyArchiverFixture:
    def __init__(self, memory_library: MemoryLibrary):
        self._library = memory_library

    async def archive(self, memory_id: UUID) -> None:
        if await self._library.long_term.is_archived(memory_id):
            return
        await self._library.archive(memory_id)

    async def resurrect(self, memory_id: UUID) -> MemoryAtom:
        memory = await self._library.long_term.load(memory_id)
        await self._library.revive(memory_id)
        return memory

    async def is_archived(self, memory_id: UUID) -> bool:
        return await self._library.long_term.is_archived(memory_id)

    async def get_archive_record(self, memory_id: UUID) -> Optional[ArchiveRecord]:
        records = await self._library.long_term.query(limit=100)
        return next((record for record in records if record.memory_id == memory_id), None)


# ========== Mock Storage ==========

class MockQdrantMemoryStore:
    """
    模拟 Qdrant 存储

    用于测试 Lifecycle 组件，无需真实数据库连接。
    """

    def __init__(self):
        self.memories: Dict[UUID, MemoryAtom] = {}
        self._call_log: List[Dict] = []

    async def get_memory(self, memory_id: UUID) -> Optional[MemoryAtom]:
        """获取记忆"""
        self._call_log.append({"method": "get_memory", "memory_id": memory_id})
        return self.memories.get(memory_id)

    async def upsert_memory(self, memory: MemoryAtom) -> None:
        """插入或更新记忆"""
        self._call_log.append({"method": "upsert_memory", "memory_id": memory.id})
        self.memories[memory.id] = memory

    async def delete_memory(self, memory_id: UUID) -> bool:
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
        mid_term=MidTermMemoryStore(primary=_MockMidTermAdapter(mock_storage)),
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
def archiver(mock_storage, archiver_config) -> _LegacyArchiverFixture:
    """提供归档器实例"""
    memory_library = MemoryLibrary(
        short_term=ShortTermMemoryStore(),
        mid_term=MidTermMemoryStore(primary=_MockMidTermAdapter(mock_storage)),
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(
                archive_dir=archiver_config.archive_dir,
                compress=archiver_config.compression,
            )
        ),
    )
    return _LegacyArchiverFixture(memory_library)


# ========== 测试类: 评分逻辑 ==========

class TestVitalityScoring:
    """
    评分逻辑测试

    验证 VitalityCalculator 的评分公式:
    V = (C × I) × D(t) × 100 + A
    """

    def test_lif_scr_001_base_score_by_type(self, vitality_calculator):
        """
        LIF-SCR-001: 基础分计算 (三段式语义)

        新公式 V = V_0·D(t) + A + B 中，V_0 固定，类型差异通过 λ_eff 调制衰减。
        本测试改为验证: 30 天后，CODE_SNIPPET (I=1.0, λ_eff=λ) 衰减比
        WORK_IN_PROGRESS (I=0.5, λ_eff=1.5λ) 更慢，故分数更高。
        """
        case = get_scoring_test_by_id("LIF-SCR-001")
        print_test_header(case["id"], case["name"])

        # 创建两种 30 天前的记忆 (相同 confidence, access_count=0)
        # 验证类型通过 λ_eff 调制衰减率造成的差异
        code_memory = create_memory_with_age(
            days_old=30,
            template_name="code_snippet",
            confidence_score=0.9,
            access_count=0,
        )
        wip_memory = create_memory_with_age(
            days_old=30,
            template_name="work_in_progress",
            confidence_score=0.9,
            access_count=0,
        )

        code_score = vitality_calculator.calculate(code_memory)
        wip_score = vitality_calculator.calculate(wip_memory)

        assert code_score > wip_score, (
            f"After 30 days, CODE_SNIPPET ({code_score:.2f}) should decay slower than "
            f"WORK_IN_PROGRESS ({wip_score:.2f}) due to lower λ_eff"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"30d后 CODE={code_score:.2f}, WIP={wip_score:.2f}"
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

        # 预期衰减因子 (三段式: λ_eff = λ * (2 - I), FACT 的 I=0.9)
        # 新公式 V_0·D(t) + A + B；fresh 与 old 都无 A/B，故 ratio = D(30)/D(0) = D(30)
        fact_intrinsic = 0.9  # FACT 默认权重
        lambda_eff = 0.01 * (2.0 - fact_intrinsic)  # = 0.011
        expected_decay = math.exp(-lambda_eff * 30)  # ≈ 0.7189
        tolerance = case["tolerance"]

        # 验证衰减
        assert old_score < fresh_score, (
            f"Old memory ({old_score:.2f}) should have lower score than "
            f"fresh memory ({fresh_score:.2f})"
        )

        # 验证衰减比例接近预期 (按三段式 λ_eff 计算)
        actual_ratio = old_score / fresh_score if fresh_score > 0 else 0
        assert abs(actual_ratio - expected_decay) < tolerance, (
            f"Decay ratio ({actual_ratio:.4f}) should be close to "
            f"expected ({expected_decay:.4f}) under λ_eff={lambda_eff}"
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


# 测试类: 强化事件 ==========

class TestReinforcement:
    """
    强化事件测试

    验证 DynamicReinforcementEngine 对各种事件的处理。
    """

    @pytest.mark.asyncio
    async def test_lif_rnf_005_fresh_memory_hit_no_regression(
        self, mock_storage, reinforcement_engine, vitality_calculator
    ):
        """
        LIF-RNF-005: 新记忆首次 HIT 不应导致 vitality 假性下降 (回归用例)

        场景: 刚刚写入的记忆(MetaData 默认 confidence=0.6, vitality_score=100)，
              下一轮对话检索命中触发 HIT 事件。
        修复前: 100 -> 67 (公式 (C×I)×D(t)×100+A 把 confidence 压低 V_0)
        修复后: 100 -> 100 (V_0 与 confidence 解耦，事件加成进 B 项)

        三段式新契约:
        - V_0 固定 100 (与 confidence 解耦)
        - HIT 不重置 updated_at (衰减钟持续作用)
        - 事件加成 += hit_boost 累加进 event_vitality_boost (B 项)
        """
        case = get_reinforcement_test_by_id("LIF-RNF-005")
        print_test_header(case["id"], case["name"])

        # 构造刚刚写入的新记忆 (CODE_SNIPPET, intrinsic=1.0, 衰减最慢)
        # 关键: 不显式赋 confidence/vitality_score/access_count/event_vitality_boost，
        #       仿真 MemoryGenerationEngine._draft_to_memory 仅设 confidence 之外的默认路径。
        memory = create_test_memory(
            template_name="code_snippet",
            vitality_score=100.0,     # MetaData 字段默认 100，与 V_0 自洽
            confidence_score=0.6,     # MetaData 字段默认 0.6，旧公式下会被压低
            access_count=0,
        )
        # event_vitality_boost 重置为 0 (仿真刚写入、未被强化的初态)
        memory.meta.event_vitality_boost = 0.0
        # 强制更新时间戳贴近"刚刚写入"，避免 days_since(update) 含小数 >0 导致 D<1
        memory.meta.created_at = datetime.now()
        memory.meta.updated_at = datetime.now()
        pre_updated_at = memory.meta.updated_at

        await mock_storage.upsert_memory(memory)

        # 触发 HIT 事件
        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=memory.id,
            source="test_retrieval_hit",
        )
        result = await reinforcement_engine.reinforce(memory.id, event)

        updated_memory = await mock_storage.get_memory(memory.id)

        # 验证 1: vitality 保持 100 (修复前会是 67)
        assert result.previous_vitality == 100.0, (
            f"新记忆 previous_vitality 应为 100，实际 {result.previous_vitality}"
        )
        assert result.new_vitality == 100.0, (
            f"新记忆 HIT 后 new_vitality 应保持 100 (修复后)，实际 {result.new_vitality} "
            f"(修复前 bug 表现为 67)"
        )

        # 验证 2: 事件加成累加进 B 项 (event_vitality_boost)
        assert updated_memory.meta.event_vitality_boost == case["expected_event_vitality_boost"], (
            f"event_vitality_boost 应累加 hit_boost=5，实际 "
            f"{updated_memory.meta.event_vitality_boost}"
        )

        # 验证 3: HIT 不重置 updated_at (衰减钟持续作用，艾宾浩斯语义)
        if case["expected_updated_at_unchanged_on_hit"]:
            assert updated_memory.meta.updated_at == pre_updated_at, (
                "HIT 不应重置 updated_at (仅 CITATION 重置) — 让遗忘曲线持续作用"
            )

        # 验证 4: access_count += 1
        assert updated_memory.meta.access_count == 1, (
            f"access_count 应递增为 1，实际 {updated_memory.meta.access_count}"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Vitality: {result.previous_vitality:.1f} -> {result.new_vitality:.1f}, "
            f"B={updated_memory.meta.event_vitality_boost}, "
            f"updated_at unchanged={updated_memory.meta.updated_at == pre_updated_at}"
        )

    @pytest.mark.asyncio
    async def test_lif_rnf_001_hit_event(self, mock_storage, reinforcement_engine):
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
        await mock_storage.upsert_memory(memory)

        initial_vitality = memory.meta.vitality_score
        initial_access_count = memory.meta.access_count

        # 触发 HIT 事件
        event = MemoryEvent(
            event_type=EventType.HIT,
            memory_id=memory.id,
            source="test",
        )
        result = await reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = await mock_storage.get_memory(memory.id)

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

    @pytest.mark.asyncio
    async def test_lif_rnf_002_citation_event(self, mock_storage, reinforcement_engine):
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
        await mock_storage.upsert_memory(old_memory)

        # 触发 CITATION 事件
        event = MemoryEvent(
            event_type=EventType.CITATION,
            memory_id=old_memory.id,
            source="test",
        )
        result = await reinforcement_engine.reinforce(old_memory.id, event)

        # 获取更新后的记忆
        updated_memory = await mock_storage.get_memory(old_memory.id)

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

    @pytest.mark.asyncio
    async def test_lif_rnf_003_negative_feedback(self, mock_storage, reinforcement_engine):
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
        await mock_storage.upsert_memory(memory)

        initial_confidence = memory.meta.confidence_score

        # 触发负面反馈事件
        event = MemoryEvent(
            event_type=EventType.FEEDBACK_NEGATIVE,
            memory_id=memory.id,
            source="user",
        )
        result = await reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = await mock_storage.get_memory(memory.id)

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

    @pytest.mark.asyncio
    async def test_lif_rnf_004_positive_feedback(self, mock_storage, reinforcement_engine):
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
        await mock_storage.upsert_memory(memory)

        initial_access_count = memory.meta.access_count

        # 触发正面反馈事件
        event = MemoryEvent(
            event_type=EventType.FEEDBACK_POSITIVE,
            memory_id=memory.id,
            source="user",
        )
        result = await reinforcement_engine.reinforce(memory.id, event)

        # 获取更新后的记忆
        updated_memory = await mock_storage.get_memory(memory.id)

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

    验证 MemoryLibrary + LongTermMemoryStore 的归档和唤醒流程。
    """

    @pytest.mark.asyncio
    async def test_lif_arc_001_archive_trigger(self, mock_storage, archiver):
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
        await mock_storage.upsert_memory(memory)
        memory_id = memory.id

        # 验证记忆在热存储中
        assert await mock_storage.get_memory(memory_id) is not None, (
            "Memory should exist in hot storage before archive"
        )

        # 执行归档
        await archiver.archive(memory_id)

        # 验证记忆从热存储中删除
        assert await mock_storage.get_memory(memory_id) is None, (
            "Memory should be deleted from hot storage after archive"
        )

        # 验证记忆在冷存储中
        assert await archiver.is_archived(memory_id), (
            "Memory should be in cold storage after archive"
        )

        # 验证归档文件存在
        record = await archiver.get_archive_record(memory_id)
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

    @pytest.mark.asyncio
    async def test_lif_arc_002_resurrect_flow(self, mock_storage, archiver):
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
        await mock_storage.upsert_memory(original_memory)
        memory_id = original_memory.id

        # 保存原始数据用于比较
        original_title = original_memory.index.title
        original_content = original_memory.payload.content
        original_tags = original_memory.index.tags.copy()

        # 归档
        await archiver.archive(memory_id)
        assert await archiver.is_archived(memory_id), "Memory should be archived"

        # 唤醒
        resurrected_memory = await archiver.resurrect(memory_id)

        # 验证记忆回到热存储
        assert await mock_storage.get_memory(memory_id) is not None, (
            "Memory should be back in hot storage"
        )

        # 验证不再在冷存储中
        assert not await archiver.is_archived(memory_id), (
            "Memory should not be in cold storage after resurrect"
        )

        # 验证数据完整性
        assert resurrected_memory.id == memory_id, "ID should match"
        assert resurrected_memory.index.title == original_title, "Title should match"
        assert resurrected_memory.payload.content == original_content, "Content should match"
        # tags 经 IndexLayer.validate_tags 使用 set 去重，顺序非确定，改用集合比较
        assert set(resurrected_memory.index.tags) == set(original_tags), "Tags should match"

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Data integrity verified: title, content, tags all match"
        )

    @pytest.mark.asyncio
    async def test_lif_arc_003_archive_idempotency(self, mock_storage, archiver):
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
        await mock_storage.upsert_memory(memory)
        memory_id = memory.id

        # 第一次归档
        await archiver.archive(memory_id)
        assert await archiver.is_archived(memory_id), "Memory should be archived"

        # 获取归档记录
        first_record = await archiver.get_archive_record(memory_id)
        first_path = first_record.storage_path

        # 第二次归档（应该是幂等的，不产生错误）
        await archiver.archive(memory_id)

        # 验证没有重复文件
        record_after = await archiver.get_archive_record(memory_id)
        assert record_after.storage_path == first_path, (
            "Archive path should remain the same"
        )

        print_test_result(
            case["id"],
            case["name"],
            True,
            f"Idempotent: no error, no duplicate file"
        )


# ========== 主入口 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
