from unittest.mock import AsyncMock

import pytest

from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@pytest.mark.asyncio
async def test_list_memory_tasks_requests_patchouli_route():
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=["task"])
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, handler)
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.list_memory_tasks()

    # 结果经真实总线派发到达，验证 request→返回完整链路
    assert result == ["task"]
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_memory_task_requests_patchouli_route():
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value="task")
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_GET, handler)
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.get_memory_task("task_1")

    assert result == "task"
    handler.assert_awaited_once_with("task_1")


@pytest.mark.asyncio
async def test_cancel_memory_task_requests_patchouli_route():
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL, handler)
    service = MemoryTaskApplicationService(global_bus=bus)

    result = await service.cancel_memory_task("task_1")

    assert result is True
    handler.assert_awaited_once_with("task_1")
