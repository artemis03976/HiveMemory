from unittest.mock import AsyncMock

import pytest

from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.contracts.routes import GlobalRoutes


@pytest.mark.asyncio
async def test_list_memory_tasks_requests_patchouli_route():
    bus = AsyncMock()
    bus.request.return_value = ["task"]
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.list_memory_tasks()

    assert result == ["task"]
    bus.request.assert_awaited_once_with(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST)


@pytest.mark.asyncio
async def test_get_memory_task_requests_patchouli_route():
    bus = AsyncMock()
    bus.request.return_value = "task"
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.get_memory_task("task_1")

    assert result == "task"
    bus.request.assert_awaited_once_with(
        GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
        "task_1",
    )


@pytest.mark.asyncio
async def test_cancel_memory_task_requests_patchouli_route():
    bus = AsyncMock()
    bus.request.return_value = True
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.cancel_memory_task("task_1")

    assert result is True
    bus.request.assert_awaited_once_with(
        GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
        "task_1",
    )
