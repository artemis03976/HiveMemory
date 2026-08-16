"""
LifecycleFamiliar 单元测试

测试覆盖:
- run_gardening_once: 全局维护调度器入口（编排合约 + 异常处理）
- record_hit: 字符串 ID 归一化为 UUID
"""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID

from hivememory.patchouli.services.lifecycle import LifecycleFamiliar


class TestLifecycleFamiliar:
    """LifecycleFamiliar 完整测试套件"""

    def _make_familiar(self, lifecycle=None, memory_lib=None):
        lifecycle = lifecycle or Mock()
        memory_lib = memory_lib or Mock()
        return LifecycleFamiliar(
            lifecycle_engine=lifecycle,
            memory_library=memory_lib,
        )

    @pytest.mark.asyncio
    async def test_run_gardening_once_delegates_to_lifecycle_engine(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection = AsyncMock(return_value=3)
        familiar = self._make_familiar(lifecycle=lifecycle)

        result = await familiar.run_gardening_once()

        assert result["success"] is True
        assert result["archived_count"] == 3
        lifecycle.run_garbage_collection.assert_awaited_once_with(force=False)

    @pytest.mark.asyncio
    async def test_run_gardening_once_handles_exception(self):
        lifecycle = Mock()
        lifecycle.run_garbage_collection = AsyncMock(side_effect=RuntimeError("GC failed"))
        familiar = self._make_familiar(lifecycle=lifecycle)

        result = await familiar.run_gardening_once()

        assert result["success"] is False
        assert "GC failed" in result["error"]
        assert result["archived_count"] == 0

    @pytest.mark.asyncio
    async def test_normalize_uuid_converts_string_to_uuid(self):
        lifecycle = Mock()
        lifecycle.record_hit = AsyncMock()
        familiar = self._make_familiar(lifecycle=lifecycle)
        uuid_obj = uuid4()
        str_id = str(uuid_obj)

        await familiar.record_hit(str_id)

        call_args = lifecycle.record_hit.call_args
        assert isinstance(call_args[0][0], UUID)
        assert str(call_args[0][0]) == str_id
