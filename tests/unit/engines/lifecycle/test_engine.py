from unittest.mock import AsyncMock, Mock, PropertyMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
from hivememory.engines.lifecycle.models import (
    EventType,
    MemoryEvent,
    ReinforcementResult,
)


def _make_memory(vitality_score=50.0) -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(
            source_agent_id="a1",
            user_id="u1",
            session_id="s1",
            vitality_score=vitality_score,
        ),
        index=IndexLayer(
            title="Test",
            summary="A long enough test summary for validation",
            tags=["t"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="Content"),
    )


def _make_engine():
    mid_term = AsyncMock()
    vitality = Mock()
    reinforcement = AsyncMock()
    gc = Mock()
    gc.collect = AsyncMock()
    engine = MemoryLifecycleEngine(
        mid_term=mid_term,
        vitality_calculator=vitality,
        reinforcement_engine=reinforcement,
        garbage_collector=gc,
    )
    return engine, mid_term, vitality, reinforcement, gc


class TestLifecycleEngineVitality:
    def setup_method(self):
        (
            self.engine,
            self.mock_mid_term,
            self.mock_vitality,
            self.mock_reinforcement,
            self.mock_gc,
        ) = _make_engine()

    @pytest.mark.asyncio
    async def test_refresh_vitality_no_persist(self):
        mem = _make_memory(vitality_score=10.0)
        self.mock_vitality.calculate.return_value = 72.0

        result = await self.engine.refresh_vitality(mem, persist=False)

        assert mem.meta.vitality_score == pytest.approx(72.0)
        self.mock_mid_term.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_vitality_persist(self):
        mem = _make_memory(vitality_score=10.0)
        self.mock_vitality.calculate.return_value = 72.0

        result = await self.engine.refresh_vitality(mem, persist=True)

        assert mem.meta.vitality_score == pytest.approx(72.0)
        self.mock_mid_term.upsert.assert_awaited_once_with(mem)

    @pytest.mark.asyncio
    async def test_refresh_vitality_batch(self):
        m1 = _make_memory()
        m2 = _make_memory()
        self.mock_vitality.calculate.side_effect = [30.0, 80.0]

        await self.engine.refresh_vitality_batch([m1, m2], persist=True)

        assert m1.meta.vitality_score == pytest.approx(30.0)
        assert m2.meta.vitality_score == pytest.approx(80.0)
        upserted = [call.args[0] for call in self.mock_mid_term.upsert.await_args_list]
        assert upserted == [m1, m2]


class TestLifecycleEngineEvents:
    def setup_method(self):
        (
            self.engine,
            self.mock_mid_term,
            self.mock_vitality,
            self.mock_reinforcement,
            self.mock_gc,
        ) = _make_engine()
        self.mock_reinforcement.reinforce.return_value = AsyncMock(
            spec=ReinforcementResult
        )

    @pytest.mark.asyncio
    async def test_record_hit(self):
        mid = uuid4()
        await self.engine.record_hit(mid, source="retrieval")

        event = self.mock_reinforcement.reinforce.call_args[0][1]
        assert event.event_type == EventType.HIT
        assert event.memory_id == mid
        assert event.source == "retrieval"

    @pytest.mark.asyncio
    async def test_record_citation(self):
        mid = uuid4()
        await self.engine.record_citation(mid, source="agent")

        event = self.mock_reinforcement.reinforce.call_args[0][1]
        assert event.event_type == EventType.CITATION
        assert event.memory_id == mid
        assert event.source == "agent"

    @pytest.mark.asyncio
    async def test_record_feedback_positive(self):
        mid = uuid4()
        await self.engine.record_feedback(mid, positive=True)

        event = self.mock_reinforcement.reinforce.call_args[0][1]
        assert event.event_type == EventType.FEEDBACK_POSITIVE

    @pytest.mark.asyncio
    async def test_record_feedback_negative(self):
        mid = uuid4()
        await self.engine.record_feedback(mid, positive=False)

        event = self.mock_reinforcement.reinforce.call_args[0][1]
        assert event.event_type == EventType.FEEDBACK_NEGATIVE

    @pytest.mark.asyncio
    async def test_record_event_delegates(self):
        mid = uuid4()
        event = MemoryEvent(event_type=EventType.HIT, memory_id=mid, source="test")

        await self.engine.record_event(event)

        self.mock_reinforcement.reinforce.assert_called_once_with(mid, event)


class TestLifecycleEngineDelegation:
    def setup_method(self):
        (
            self.engine,
            self.mock_mid_term,
            self.mock_vitality,
            self.mock_reinforcement,
            self.mock_gc,
        ) = _make_engine()

    @pytest.mark.asyncio
    async def test_run_garbage_collection_refreshes_before_collect(self):
        m1 = _make_memory(vitality_score=10.0)
        m2 = _make_memory(vitality_score=90.0)
        self.mock_mid_term.scroll.return_value = [m1, m2]
        self.mock_vitality.calculate.side_effect = [12.0, 88.0]
        self.mock_gc.collect.return_value = 3

        result = await self.engine.run_garbage_collection(force=True)

        self.mock_mid_term.scroll.assert_awaited_once_with(limit=10000)
        assert self.mock_mid_term.upsert.await_count == 2
        assert m1.meta.vitality_score == pytest.approx(12.0)
        assert m2.meta.vitality_score == pytest.approx(88.0)
        self.mock_gc.collect.assert_awaited_once_with([m1, m2], force=True)


class TestLifecycleEngineQueries:
    def setup_method(self):
        (
            self.engine,
            self.mock_mid_term,
            self.mock_vitality,
            self.mock_reinforcement,
            self.mock_gc,
        ) = _make_engine()

    @pytest.mark.asyncio
    async def test_get_low_vitality_memories(self):
        m1 = _make_memory()
        m2 = _make_memory()
        m3 = _make_memory()
        self.mock_mid_term.scroll.return_value = [m1, m2, m3]
        self.mock_vitality.calculate.side_effect = [10.0, 50.0, 5.0]

        results = await self.engine.get_low_vitality_memories(threshold=20.0)

        assert len(results) == 2
        assert results[0][1] == 5.0
        assert results[1][1] == 10.0

    @pytest.mark.asyncio
    async def test_get_low_vitality_memories_with_limit(self):
        m1 = _make_memory()
        m2 = _make_memory()
        self.mock_mid_term.scroll.return_value = [m1, m2]
        self.mock_vitality.calculate.side_effect = [5.0, 10.0]

        results = await self.engine.get_low_vitality_memories(threshold=20.0, limit=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_low_vitality_memories_none_below_threshold(self):
        m1 = _make_memory()
        self.mock_mid_term.scroll.return_value = [m1]
        self.mock_vitality.calculate.return_value = 90.0

        results = await self.engine.get_low_vitality_memories(threshold=20.0)

        assert results == []

    def test_get_event_history_supported(self):
        mock_history = [Mock(spec=ReinforcementResult)]
        self.mock_reinforcement.get_event_history = Mock(return_value=mock_history)

        self.engine.get_event_history(limit=50)

        self.mock_reinforcement.get_event_history.assert_called_once_with(
            None,
            50,
        )

    def test_get_event_history_unsupported(self):
        del self.mock_reinforcement.get_event_history

        results = self.engine.get_event_history()

        assert results == []

    def test_get_stats(self):
        self.mock_gc.get_stats.return_value = {"collected": 5}
        self.mock_reinforcement.get_stats = Mock(return_value={"events": 10})

        stats = self.engine.get_stats()

        # 组装结构契约：各子系统统计被聚合到对应 key（值来自子系统自身）
        assert "garbage_collector" in stats
        assert "reinforcement" in stats

    def test_get_stats_without_optional_methods(self):
        del self.mock_gc.get_stats
        del self.mock_reinforcement.get_stats

        stats = self.engine.get_stats()

        assert "garbage_collector" in stats
