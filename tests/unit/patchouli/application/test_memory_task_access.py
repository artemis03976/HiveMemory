"""MemoryTaskManagementService 授权与归属校验的单元测试。

被测对象（父计划 5.6.4/5.7.1，WRX-1 冻结）：
- get/wait/list 绑定 ``task.observe``，cancel 绑定 ``management.task``；
- 提供 access 时任务必须携带归属投影且与上下文一致：跨 scope、无归属与
  不存在统一按 not found 拒绝，不泄漏其他 Workspace 的任务存在性；
- 无 access 的旧调用为迁移期受信适配，保持既有行为。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import (
    OperationDeniedError,
    ResourceNotFoundError,
)
from hivememory.patchouli.application import MemoryTaskManagementService
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class FakeTaskBus:
    """以内存字典承载任务的路由假总线（控制器为边界外协作者）。"""

    def __init__(self, tasks):
        self._tasks = {task.task_id: task for task in tasks}

    async def request(self, route, *args, **kwargs):
        if route == "memory_task.get":
            return self._tasks.get(args[0])
        if route == "memory_task.cancel":
            return args[0] in self._tasks
        if route == "memory_task.list":
            return list(self._tasks.values())
        raise AssertionError(f"unexpected route: {route}")


def _task(task_id="active:i1", workspace=MAIN, *, scoped=True):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id="topic_1",
        label="topic_1",
        source=MemoryGenerationSource.WRITE,
        pending_alias="draft_x_0001",
        status=MemoryGenerationTaskStatus.PENDING,
        identity_scope=(
            make_identity_scope(
                user_id=workspace.owner_user_id,
                agent_id="a1",
                workspace_id=workspace.workspace_id,
            )
            if scoped
            else None
        ),
        submitted_by="local-process:test" if scoped else None,
    )


def _run(coro):
    return asyncio.run(coro)


async def _context(operation, workspace=MAIN):
    admission = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    actor = make_identity_scope(
        user_id="u1", agent_id="a1", workspace_id=workspace.workspace_id
    ).actor_identity
    return await admission.admit(CallerPrincipal("local-process:test"), actor, workspace, operation)


def test_observe_with_matching_scope_returns_task_projection():
    """task.observe + 归属一致：返回任务只读投影。"""
    task = _task()
    service = MemoryTaskManagementService(bus=FakeTaskBus([task]))
    context = _run(_context(WorkspaceOperation.TASK_OBSERVE))

    result = _run(service.get_memory_task(task.task_id, access=context))

    assert result is not None
    assert result.identity_scope == context.identity_scope


def test_observe_rejects_cross_scope_and_unscoped_tasks_as_not_found():
    """跨 scope 与无归属（legacy）任务统一 not found，不泄漏存在性。"""
    foreign = _task(workspace=OTHER)
    legacy = _task(task_id="active:legacy", scoped=False)
    service = MemoryTaskManagementService(bus=FakeTaskBus([foreign, legacy]))
    context = _run(_context(WorkspaceOperation.TASK_OBSERVE))

    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task(foreign.task_id, access=context))
    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task(legacy.task_id, access=context))
    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task("active:ghost", access=context))

    # 无 access 的旧调用保持既有行为（受信适配）
    assert _run(service.get_memory_task(foreign.task_id)) is foreign


def test_cancel_requires_management_task_grant_not_task_observe():
    """task.observe 不授予取消：取消绑定 management.task。"""
    task = _task()
    service = MemoryTaskManagementService(bus=FakeTaskBus([task]))
    observe_context = _run(_context(WorkspaceOperation.TASK_OBSERVE))

    with pytest.raises(OperationDeniedError):
        _run(service.cancel_memory_task(task.task_id, access=observe_context))


def test_cancel_with_management_grant_checks_ownership():
    """management.task 取消自己的任务放行；跨 scope 拒绝。"""
    task = _task()
    service = MemoryTaskManagementService(bus=FakeTaskBus([task]))
    management_context = _run(_context(WorkspaceOperation.MANAGEMENT_TASK))

    assert _run(service.cancel_memory_task(task.task_id, access=management_context)) is True

    foreign = _task(task_id="active:foreign", workspace=OTHER)
    other_scope_context = _run(_context(WorkspaceOperation.MANAGEMENT_TASK, workspace=OTHER))
    service_own = MemoryTaskManagementService(bus=FakeTaskBus([foreign]))
    with pytest.raises(ResourceNotFoundError):
        _run(
            service_own.cancel_memory_task(
                task.task_id,
                access=other_scope_context,
            )
        )


def test_list_with_access_filters_to_own_workspace():
    """带 access 的列表按归属过滤；legacy 列表保持全量行为。"""
    own = _task()
    foreign = _task(task_id="active:i2", workspace=OTHER)
    service = MemoryTaskManagementService(bus=FakeTaskBus([own, foreign]))
    context = _run(_context(WorkspaceOperation.TASK_OBSERVE))

    filtered = _run(service.list_memory_tasks(access=context))
    assert [t.task_id for t in filtered] == [own.task_id]

    everything = _run(service.list_memory_tasks())
    assert len(everything) == 2
