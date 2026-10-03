"""记忆生成任务能力（``workspace.capability.memory_tasks``）委托测试。

观察/取消的 operation 授权在本层、路由调用前执行（A1 访问边界返工第
4.3 节）：缺少许可的 context 以 ``OperationDeniedError`` 拒绝，不触达
Patchouli 路由。
"""

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
)


@pytest.fixture
def composition():
    """管理入口语义的组合：system 记录只授观察与管理任务。"""
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="system",
                allowed_operations=frozenset(
                    {WorkspaceOperation.TASK_OBSERVE, WorkspaceOperation.MANAGEMENT_TASK}
                ),
            )
        ]
    )


def _service(bus: GlobalSystemBus, composition) -> MemoryTaskApplicationService:
    return MemoryTaskApplicationService(global_bus=bus, access_guard=composition.guard)


@pytest.mark.asyncio
async def test_list_memory_tasks_requests_patchouli_route(composition):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=["task"])
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).list_memory_tasks(access=access)

    # 结果经真实总线派发到达，验证 request→返回完整链路
    assert result == ["task"]
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_memory_task_requests_patchouli_route(composition):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value="task")
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_GET, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).get_memory_task("task_1", access=access)

    assert result == "task"
    handler.assert_awaited_once_with("task_1", access=access)


@pytest.mark.asyncio
async def test_cancel_memory_task_requests_patchouli_route(composition):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).cancel_memory_task("task_1", access=access)

    assert result is True
    handler.assert_awaited_once_with("task_1", access=access)


@pytest.mark.asyncio
async def test_missing_task_observe_operation_rejected_before_route(composition):
    """context 缺少 ``task.observe`` 时在能力层拒绝，不触达 Patchouli 路由。"""
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=["task"])
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, handler)
    composition_no_observe = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="system",
                allowed_operations=frozenset(),
            )
        ]
    )
    access = await composition_no_observe.authenticate(agent_id="system", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        await _service(bus, composition_no_observe).list_memory_tasks(access=access)

    assert exc_info.value.details["operation"] == WorkspaceOperation.TASK_OBSERVE.value
    handler.assert_not_awaited()
