"""
MemoryTaskManagementService 单元测试

测试覆盖:
- list_memory_tasks: 列出所有内存任务
- get_memory_task: 获取单个任务
- cancel_memory_task: 取消任务
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from hivememory.patchouli.application.memory_task_management_service import MemoryTaskManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class TestMemoryTaskManagementService:
    """MemoryTaskManagementService 测试套件"""

    def _make_service(self, bus=None):
        if bus is None:
            bus = Mock()
        return MemoryTaskManagementService(bus=bus)

    @pytest.mark.asyncio
    async def test_list_memory_tasks_delegates_to_bus(self):
        expected_tasks = ["task1", "task2"]
        bus = Mock()
        bus.request = AsyncMock(return_value=expected_tasks)

        service = MemoryTaskManagementService(bus=bus)
        result = await service.list_memory_tasks()

        assert result == expected_tasks
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_TASK_LIST)

    @pytest.mark.asyncio
    async def test_get_memory_task_delegates_to_bus(self):
        task_id = "task_123"
        expected_task = Mock()
        bus = Mock()
        bus.request = AsyncMock(return_value=expected_task)

        service = MemoryTaskManagementService(bus=bus)
        result = await service.get_memory_task(task_id)

        assert result is expected_task
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)

    @pytest.mark.asyncio
    async def test_get_memory_task_returns_none_when_not_found(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=None)

        service = MemoryTaskManagementService(bus=bus)
        result = await service.get_memory_task("missing_task")

        assert result is None
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_TASK_GET, "missing_task")

    @pytest.mark.asyncio
    async def test_cancel_memory_task_delegates_to_bus(self):
        task_id = "task_123"
        bus = Mock()
        bus.request = AsyncMock(return_value=True)

        service = MemoryTaskManagementService(bus=bus)
        result = await service.cancel_memory_task(task_id)

        assert result is True
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)

    @pytest.mark.asyncio
    async def test_cancel_memory_task_returns_false_when_task_not_found(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=False)

        service = MemoryTaskManagementService(bus=bus)
        result = await service.cancel_memory_task("missing_task")

        assert result is False
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, "missing_task")
