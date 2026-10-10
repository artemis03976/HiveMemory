"""root 记忆生成任务查询与取消的公共路由契约测试。"""

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
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
    """管理入口语义的组合：system 记录只提供准入，不配置 actor 操作。"""
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="system",
                allowed_operations=frozenset(),
            )
        ],
        default_workspace=workspace,
    )


def _service(bus: GlobalSystemBus, composition) -> RootMemoryTaskService:
    return RootMemoryTaskService(global_bus=bus, operation_authorizer=composition.authorizer)


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
    expected_scope = composition.authorizer.authorize_root(access, workspace)
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
    expected_scope = composition.authorizer.authorize_root(access, workspace)
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
    expected_scope = composition.authorizer.authorize_root(access, workspace)
    handler.assert_awaited_once_with("task_1", identity_scope=expected_scope)
