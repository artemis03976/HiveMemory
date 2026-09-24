"""
HiveMemory - 强化引擎单元测试

测试内容:
- HIT 事件处理
- CITATION 事件处理
- FEEDBACK 事件处理 (正面/负面)
- 置信度调整
- 事件历史跟踪
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.engines.lifecycle.models import EventType, MemoryEvent
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.system.config import ReinforcementEngineConfig
from hivememory.utils.time import utc_now
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _identity_scope():
    return make_identity_scope(user_id="user1", agent_id="agent1")


class TestDynamicReinforcementEngine:
    """测试动态强化引擎"""

    def setup_method(self):
        """测试初始化"""
        self.mock_mid_term = AsyncMock()
        self.mock_vitality_calc = Mock()

        self.config = ReinforcementEngineConfig(
            enable_event_history=True,
        )

        self.engine = DynamicReinforcementEngine(
            mid_term=self.mock_mid_term,
            vitality_calculator=self.mock_vitality_calc,
            config=self.config,
        )

        # 创建测试记忆
        self.test_memory = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=50.0,  # 50/100
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

    def _submitted_patch(self) -> dict:
        """取 reinforce 提交给 mid_term.patch_payload 的 patch mapping（MVL-2 契约）。"""
        assert (
            self.mock_mid_term.patch_payload.call_count == 1
        ), "reinforce 应恰好提交一次 patch_payload（不再整原子 upsert）"
        args = self.mock_mid_term.patch_payload.call_args[0]
        expected_key = WorkspaceMemoryKey.from_identity_scope(
            _identity_scope(), self.test_memory.id
        )
        assert args[0] == expected_key, "patch_payload 应使用 identity_scope + memory_id 构造的键"
        return args[1]

    @pytest.mark.asyncio
    async def test_hit_event(self):
        """测试 HIT 事件累加进 event_vitality_boost (B 项)，不推进衰减基准"""
        original_decay_anchor_at = self.test_memory.meta.lifecycle.decay_anchor_at
        original_updated_at = self.test_memory.meta.updated_at

        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 50.0  # 重算结果 (含 B 项)

        event = MemoryEvent(event_type=EventType.HIT, memory_id=self.test_memory.id, source="test")

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        assert result.event_type == EventType.HIT
        assert result.previous_vitality == 50.0
        # 事件加成累加进 B 项 (meta.lifecycle.event_vitality_boost)
        patch = self._submitted_patch()
        assert patch["meta.lifecycle.event_vitality_boost"] == self.config.hit_boost
        # HIT patch 只含 4 个基础授权字段，不提交 decay_anchor_at
        assert set(patch) == {
            "meta.lifecycle.access_count",
            "meta.lifecycle.last_accessed_at",
            "meta.lifecycle.event_vitality_boost",
            "meta.lifecycle.vitality_score",
        }
        # HIT 不推进 decay_anchor_at，也不改写内容时间 updated_at
        assert self.test_memory.meta.lifecycle.decay_anchor_at == original_decay_anchor_at
        assert self.test_memory.meta.updated_at == original_updated_at

    @pytest.mark.asyncio
    async def test_citation_resets_decay(self):
        """测试 CITATION 事件推进衰减基准，且不改写内容时间 updated_at"""
        original_updated_at = self.test_memory.meta.updated_at
        original_decay_anchor_at = self.test_memory.meta.lifecycle.decay_anchor_at

        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 70.0  # 提升效果

        event = MemoryEvent(
            event_type=EventType.CITATION, memory_id=self.test_memory.id, source="test"
        )

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        assert result.event_type == EventType.CITATION

        # A2-P 契约: CITATION 推进 meta.lifecycle.decay_anchor_at，不再更新 updated_at
        patch = self._submitted_patch()
        assert patch["meta.lifecycle.decay_anchor_at"] > original_decay_anchor_at
        assert "meta.updated_at" not in patch and "meta.lifecycle.updated_at" not in patch
        assert self.test_memory.meta.updated_at == original_updated_at

    @pytest.mark.asyncio
    async def test_negative_feedback_reduces_confidence(self):
        """测试负面反馈降低置信度"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 25.0  # 降低后

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_NEGATIVE, memory_id=self.test_memory.id, source="user"
        )

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        assert result.event_type == EventType.FEEDBACK_NEGATIVE
        assert result.new_confidence < result.previous_confidence
        # 应该降低 50%
        assert abs(result.new_confidence - 0.4) < 0.01  # 0.8 * 0.5

    @pytest.mark.asyncio
    async def test_positive_feedback_increases_vitality(self):
        """测试正面反馈增加生命力"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 100.0  # 大幅提升

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_POSITIVE, memory_id=self.test_memory.id, source="user"
        )

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        assert result.event_type == EventType.FEEDBACK_POSITIVE
        assert result.new_vitality > result.previous_vitality

    @pytest.mark.asyncio
    async def test_negative_feedback_applies_vitality_penalty_after_recalculate(self):
        """负面反馈: -50 累加进 B 项，confidence ×0.5"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 80.0  # 重算结果 (含 B 项)

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_NEGATIVE,
            memory_id=self.test_memory.id,
            source="user",
        )

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        # 新契约: 最终分数即 calculator 重算结果，不再 +adjustment
        assert result.new_vitality == 80.0
        assert result.new_confidence == pytest.approx(0.4)
        # 事件惩罚累加进 B 项 (meta.lifecycle.event_vitality_boost)
        patch = self._submitted_patch()
        assert patch["meta.lifecycle.event_vitality_boost"] == (
            self.config.negative_feedback_penalty
        )
        # FEEDBACK_NEGATIVE 额外提交 confidence_score（×0.5 后的完整值）
        assert patch["meta.lifecycle.confidence_score"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_reinforcement_clamps_vitality_to_valid_range(self):
        """测试 reinforce 内的 _clamp_vitality 将 >100 的重算结果限制到 100"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 150.0  # 重算超出范围

        event = MemoryEvent(
            event_type=EventType.FEEDBACK_POSITIVE,
            memory_id=self.test_memory.id,
            source="user",
        )

        result = await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        # _clamp_vitality 应将 150 限制到 100
        assert result.new_vitality == 100.0

    @pytest.mark.asyncio
    async def test_memory_not_found(self):
        """测试记忆不存在时抛出异常"""
        self.mock_mid_term.get_by_key.return_value = None

        event = MemoryEvent(event_type=EventType.HIT, memory_id=uuid4(), source="test")

        with pytest.raises(ValueError):
            await self.engine.reinforce(_identity_scope(), uuid4(), event)

    @pytest.mark.asyncio
    async def test_access_count_increments(self):
        """测试访问计数增加"""
        original_count = self.test_memory.meta.lifecycle.access_count

        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(event_type=EventType.HIT, memory_id=self.test_memory.id, source="test")

        await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        # 获取更新的记忆
        patch = self._submitted_patch()
        assert patch["meta.lifecycle.access_count"] == original_count + 1

    @pytest.mark.asyncio
    async def test_last_accessed_at_updated(self):
        """测试最后访问时间更新"""
        before = utc_now()

        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(event_type=EventType.HIT, memory_id=self.test_memory.id, source="test")

        await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        # 获取更新的记忆
        patch = self._submitted_patch()
        assert patch["meta.lifecycle.last_accessed_at"] >= before

    @pytest.mark.asyncio
    async def test_event_history_tracked(self):
        """测试事件历史跟踪"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(event_type=EventType.HIT, memory_id=self.test_memory.id, source="test")

        await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)

        history = self.engine.get_event_history()
        assert len(history) == 1
        assert history[0].event_type == EventType.HIT

    @pytest.mark.asyncio
    async def test_event_history_filtered_by_memory(self):
        """测试按记忆ID过滤历史"""
        memory1_id = uuid4()
        memory2_id = uuid4()

        # 创建两个记忆
        memory1 = MemoryAtom(
            id=memory1_id,
            meta=make_memory_metadata(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=50.0,
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
            meta=make_memory_metadata(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=50.0,
            ),
            index=IndexLayer(
                title="Test2",
                summary="Test summary 2",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

        self.mock_mid_term.get_by_key.side_effect = [memory1, memory2]
        self.mock_vitality_calc.calculate.return_value = 55.0

        # 记录两个事件
        event1 = MemoryEvent(event_type=EventType.HIT, memory_id=memory1_id, source="test")
        event2 = MemoryEvent(event_type=EventType.HIT, memory_id=memory2_id, source="test")

        await self.engine.reinforce(_identity_scope(), memory1_id, event1)
        await self.engine.reinforce(_identity_scope(), memory2_id, event2)

        # 过滤 memory1 的事件
        history = self.engine.get_event_history(memory_id=memory1_id)
        assert len(history) == 1
        assert history[0].memory_id == memory1_id

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """测试清空历史"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        event = MemoryEvent(event_type=EventType.HIT, memory_id=self.test_memory.id, source="test")

        await self.engine.reinforce(_identity_scope(), self.test_memory.id, event)
        assert len(self.engine.get_event_history()) == 1

        self.engine.clear_history()
        assert len(self.engine.get_event_history()) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """测试获取统计信息"""
        self.mock_mid_term.get_by_key.return_value = self.test_memory
        self.mock_vitality_calc.calculate.return_value = 55.0

        # 记录多个事件
        for i in range(3):
            event = MemoryEvent(
                event_type=EventType.HIT, memory_id=self.test_memory.id, source=f"test{i}"
            )
            await self.engine.reinforce(
                _identity_scope(),
                self.test_memory.id,
                event,
            )

        stats = self.engine.get_stats()
        assert stats["total_events"] == 3
        assert "event_counts" in stats
