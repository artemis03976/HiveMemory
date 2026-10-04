"""记忆生成任务能力（``workspace.capability.memory_tasks``）观察/取消测试。

观察/取消的 operation 授权在本层、路由调用前执行（A1 访问边界返工第 4.5
节）：白名单缺少对应 operation 的 context 以 ``OperationDeniedError`` 拒绝，
不触达 Patchouli 路由；获准时向 Patchouli 路由传 guard 组装的
``identity_scope``，不再传 access context。
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
    make_workspace_identity,
)


@pytest.fixture
def workspace():
    """观察/取消的目标 workspace（等于 context 的驻留 workspace）。"""
    return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


@pytest.fixture
def composition(workspace):
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
        ],
        default_workspace=workspace,
    )


def _service(bus: GlobalSystemBus, composition) -> MemoryTaskApplicationService:
    return MemoryTaskApplicationService(global_bus=bus, operation_authorizer=composition.authorizer)


@pytest.mark.asyncio
async def test_list_memory_tasks_requests_patchouli_route(composition, workspace):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=["task"])
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).list_memory_tasks(
        target_workspace=workspace, access=access
    )

    # 结果经真实总线派发到达，验证 request→返回完整链路
    assert result == ["task"]
    handler.assert_awaited_once()
    expected_scope = composition.authorizer.authorize_operation(
        access, WorkspaceOperation.TASK_OBSERVE, workspace
    )
    assert handler.await_args.kwargs["identity_scope"] == expected_scope
    assert "access" not in handler.await_args.kwargs


@pytest.mark.asyncio
async def test_get_memory_task_requests_patchouli_route(composition, workspace):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value="task")
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_GET, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).get_memory_task(
        "task_1", target_workspace=workspace, access=access
    )

    assert result == "task"
    expected_scope = composition.authorizer.authorize_operation(
        access, WorkspaceOperation.TASK_OBSERVE, workspace
    )
    handler.assert_awaited_once_with("task_1", identity_scope=expected_scope)


@pytest.mark.asyncio
async def test_cancel_memory_task_requests_patchouli_route(composition, workspace):
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL, handler)
    access = await composition.authenticate(agent_id="system", user_id="u1")

    result = await _service(bus, composition).cancel_memory_task(
        "task_1", target_workspace=workspace, access=access
    )

    assert result is True
    expected_scope = composition.authorizer.authorize_operation(
        access, WorkspaceOperation.MANAGEMENT_TASK, workspace
    )
    handler.assert_awaited_once_with("task_1", identity_scope=expected_scope)


@pytest.mark.asyncio
async def test_missing_task_observe_operation_rejected_before_route(composition, workspace):
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
        ],
        default_workspace=workspace,
    )
    access = await composition_no_observe.authenticate(agent_id="system", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        await _service(bus, composition_no_observe).list_memory_tasks(
            target_workspace=workspace, access=access
        )

    assert exc_info.value.details["operation"] == WorkspaceOperation.TASK_OBSERVE.value
    assert exc_info.value.details["reason"] == "operation_not_allowed"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_observe_without_management_task_cannot_cancel(workspace):
    """白名单只有 ``task.observe`` 的 context 取消任务被拒，不触达 Patchouli 路由。"""
    bus = GlobalSystemBus()
    handler = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL, handler)
    observe_only = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="system",
                allowed_operations=frozenset({WorkspaceOperation.TASK_OBSERVE}),
            )
        ],
        default_workspace=workspace,
    )
    access = await observe_only.authenticate(agent_id="system", user_id="u1")

    with pytest.raises(OperationDeniedError) as exc_info:
        await _service(bus, observe_only).cancel_memory_task(
            "task_1", target_workspace=workspace, access=access
        )

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == WorkspaceOperation.MANAGEMENT_TASK.value
    handler.assert_not_awaited()
