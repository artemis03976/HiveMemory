"""
LifecycleFamiliar 单元测试

测试覆盖:
- run_gardening_once: 全局维护调度器入口
- refresh_memory_vitality: 批量生命力刷新
- record_hit: 命中事件记录
- record_citation: 引用事件记录
- record_feedback: 用户反馈记录
- revive_memory: 长期记忆复活
"""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID

from hivememory.patchouli.services.lifecycle import LifecycleFamiliar


class TestLifecycleFamiliar:
    """LifecycleFamiliar 完整测试套件"""

    def _make_familiar(self, lifecycle=None, memory_lib=None):
        return LifecycleFamiliar(
            lifecycle_engine=lifecycle,
            memory_library=memory_lib,
        )

    @pytest.mark.asyncio
    async def test_run_gardening_once_delegates_to_lifecycle_engine(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection.return_value = 3
        familiar = self._make_familiar(lifecycle=lifecycle)

        result = await familiar.run_gardening_once()

        assert result["success"] is True
        assert result["archived_count"] == 3
        lifecycle.run_garbage_collection.assert_called_once_with(force=False)

    @pytest.mark.asyncio
    async def test_run_gardening_once_reports_unavailable_engine(self):
        familiar = self._make_familiar(lifecycle=None)

        result = await familiar.run_gardening_once()

        assert result["success"] is False
        assert result["error"] == "lifecycle_engine is not available"

    @pytest.mark.asyncio
    async def test_run_gardening_once_handles_exception(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection.side_effect = RuntimeError("GC failed")
        familiar = self._make_familiar(lifecycle=lifecycle)

        result = await familiar.run_gardening_once()

        assert result["success"] is False
        assert "GC failed" in result["error"]
        assert result["archived_count"] == 0

    @pytest.mark.asyncio
    async def test_refresh_memory_vitality_delegates_to_engine(self):
        from hivememory.core.models import MemoryAtom
        lifecycle = Mock()
        lifecycle.refresh_vitality_batch = Mock(return_value=[(uuid4(), 1.0)])
        familiar = self._make_familiar(lifecycle=lifecycle)
        memory = Mock(spec=MemoryAtom)

        result = await familiar.refresh_memory_vitality([memory], persist=True)

        lifecycle.refresh_vitality_batch.assert_called_once_with([memory], persist=True)

    @pytest.mark.asyncio
    async def test_refresh_memory_vitality_raises_when_engine_unavailable(self):
        familiar = self._make_familiar(lifecycle=None)

        with pytest.raises(RuntimeError, match="lifecycle_engine is not available"):
            await familiar.refresh_memory_vitality([])

    @pytest.mark.asyncio
    async def test_record_hit_delegates_to_engine(self):
        lifecycle = Mock()
        lifecycle.record_hit = Mock(return_value={"memory_id": "id", "delta": 0.1})
        familiar = self._make_familiar(lifecycle=lifecycle)
        memory_id = uuid4()

        result = await familiar.record_hit(memory_id, source="retrieval")

        lifecycle.record_hit.assert_called_once()
        call_args = lifecycle.record_hit.call_args
        assert call_args[0][0] == memory_id
        assert call_args[1]["source"] == "retrieval"

    @pytest.mark.asyncio
    async def test_record_hit_raises_when_engine_unavailable(self):
        familiar = self._make_familiar(lifecycle=None)

        with pytest.raises(RuntimeError, match="lifecycle_engine is not available"):
            await familiar.record_hit(uuid4())

    @pytest.mark.asyncio
    async def test_record_citation_delegates_to_engine(self):
        lifecycle = Mock()
        lifecycle.record_citation = Mock(return_value={"memory_id": "id", "delta": 0.05})
        familiar = self._make_familiar(lifecycle=lifecycle)
        memory_id = uuid4()

        result = await familiar.record_citation(memory_id, source="mtp")

        lifecycle.record_citation.assert_called_once()
        call_args = lifecycle.record_citation.call_args
        assert call_args[0][0] == memory_id
        assert call_args[1]["source"] == "mtp"

    @pytest.mark.asyncio
    async def test_record_citation_raises_when_engine_unavailable(self):
        familiar = self._make_familiar(lifecycle=None)

        with pytest.raises(RuntimeError, match="lifecycle_engine is not available"):
            await familiar.record_citation(uuid4())

    @pytest.mark.asyncio
    async def test_record_feedback_delegates_to_engine(self):
        lifecycle = Mock()
        lifecycle.record_feedback = Mock(return_value={"memory_id": "id", "delta": 0.2})
        familiar = self._make_familiar(lifecycle=lifecycle)
        memory_id = uuid4()

        result = await familiar.record_feedback(memory_id, positive=True, source="ui")

        lifecycle.record_feedback.assert_called_once()
        call_args = lifecycle.record_feedback.call_args
        assert call_args[0][0] == memory_id
        assert call_args[1]["positive"] is True
        assert call_args[1]["source"] == "ui"

    @pytest.mark.asyncio
    async def test_record_feedback_raises_when_engine_unavailable(self):
        familiar = self._make_familiar(lifecycle=None)

        with pytest.raises(RuntimeError, match="lifecycle_engine is not available"):
            await familiar.record_feedback(uuid4(), positive=True, source="ui")

    @pytest.mark.asyncio
    async def test_revive_memory_calls_library_revive(self):
        memory_id = uuid4()
        memory_lib = Mock()
        memory_lib.revive = AsyncMock()
        familiar = self._make_familiar(memory_lib=memory_lib)

        await familiar.revive_memory(memory_id)

        memory_lib.revive.assert_awaited_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_revive_memory_raises_when_library_unavailable(self):
        familiar = self._make_familiar(memory_lib=None)
        memory_id = uuid4()

        with pytest.raises(RuntimeError, match="memory_library is not available"):
            await familiar.revive_memory(memory_id)

    @pytest.mark.asyncio
    async def test_normalize_uuid_converts_string_to_uuid(self):
        lifecycle = Mock()
        familiar = self._make_familiar(lifecycle=lifecycle)
        uuid_obj = uuid4()
        str_id = str(uuid_obj)

        lifecycle.record_hit = Mock()
        await familiar.record_hit(str_id)

        call_args = lifecycle.record_hit.call_args
        assert isinstance(call_args[0][0], UUID)
        assert str(call_args[0][0]) == str_id
