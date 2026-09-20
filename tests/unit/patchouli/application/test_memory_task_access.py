"""MemoryTaskManagementService 授权与归属校验的单元测试。

被测对象（A1 计划第 3.4/4.1 节）：
- get/wait/list 绑定 ``task.observe``，cancel 绑定 ``management.task``，
  行为检查先于后端读取；
- 提供 access 时任务必须携带归属投影且与上下文一致：跨 scope、无归属与
  不存在统一按 not found 拒绝，不泄漏其他 Workspace 的任务存在性；
- 无 access 的旧调用为兼容清单内的迁移期受信适配，保持既有行为。
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
from hivememory.workspace import WorkspaceOperation
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class FakeTaskBus:
    """以内存字典承载任务的路由假总线（控制器为边界外协作者）。"""

    def __init__(self, tasks):
        self._tasks = {task.task_id: task for task in tasks}
        self.cancelled: list[str] = []

    async def request(self, route, *args, **kwargs):
        if route == "memory_task.get":
            return self._tasks.get(args[0])
        if route == "memory_task.cancel":
            self.cancelled.append(args[0])
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
    """按指定 operation 构造最小许可的认证上下文与配套守卫。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                workspace_id=workspace.workspace_id,
                agent_id="a1",
                allowed_operations=frozenset({operation}),
            )
        ],
        default_workspace=workspace,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    return context, composition.guard


def _service(bus, guard):
    return MemoryTaskManagementService(bus=bus, access_guard=guard)


def test_observe_with_matching_scope_returns_task_projection():
    """task.observe + 归属一致：返回任务只读投影。"""
    task = _task()
    context, guard = _run(_context(WorkspaceOperation.TASK_OBSERVE))
    service = _service(FakeTaskBus([task]), guard)

    result = _run(service.get_memory_task(task.task_id, access=context))

    assert result is not None
    assert result.identity_scope == context.identity_scope


def test_observe_rejects_cross_scope_and_unscoped_tasks_as_not_found():
    """跨 scope 与无归属（legacy）任务统一 not found，不泄漏存在性。"""
    foreign = _task(workspace=OTHER)
    legacy = _task(task_id="active:legacy", scoped=False)
    context, guard = _run(_context(WorkspaceOperation.TASK_OBSERVE))
    service = _service(FakeTaskBus([foreign, legacy]), guard)

    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task(foreign.task_id, access=context))
    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task(legacy.task_id, access=context))
    with pytest.raises(ResourceNotFoundError):
        _run(service.get_memory_task("active:ghost", access=context))

    # 无 access 的旧调用保持既有行为（受信适配）
    assert _run(service.get_memory_task(foreign.task_id, access=None)) is foreign


def test_cancel_requires_management_operation_not_task_observe():
    """task.observe 不授予取消：取消绑定 management.task，且不触发取消副作用。"""
    task = _task()
    bus = FakeTaskBus([task])
    context, guard = _run(_context(WorkspaceOperation.TASK_OBSERVE))
    service = _service(bus, guard)

    with pytest.raises(OperationDeniedError):
        _run(service.cancel_memory_task(task.task_id, access=context))
    assert bus.cancelled == []


def test_cancel_with_management_operation_checks_ownership():
    """management.task 取消自己的任务放行；跨 scope 任务按 not found 拒绝。"""
    task = _task()
    context, guard = _run(_context(WorkspaceOperation.MANAGEMENT_TASK))
    own_bus = FakeTaskBus([task])
    assert _run(_service(own_bus, guard).cancel_memory_task(task.task_id, access=context)) is True
    assert own_bus.cancelled == [task.task_id]

    # 另一 Workspace 的 Actor 上下文：不能取消跨 scope 任务，也不暴露存在性
    foreign_context, foreign_guard = _run(
        _context(WorkspaceOperation.MANAGEMENT_TASK, workspace=OTHER)
    )
    foreign_bus = FakeTaskBus([task])
    with pytest.raises(ResourceNotFoundError):
        _run(
            _service(foreign_bus, foreign_guard).cancel_memory_task(
                task.task_id,
                access=foreign_context,
            )
        )
    assert foreign_bus.cancelled == []


def test_list_with_access_filters_to_own_workspace():
    """带 access 的列表按归属过滤；legacy 列表保持全量行为。"""
    own = _task()
    foreign = _task(task_id="active:i2", workspace=OTHER)
    context, guard = _run(_context(WorkspaceOperation.TASK_OBSERVE))
    service = _service(FakeTaskBus([own, foreign]), guard)

    filtered = _run(service.list_memory_tasks(access=context))
    assert [t.task_id for t in filtered] == [own.task_id]

    everything = _run(service.list_memory_tasks(access=None))
    assert len(everything) == 2
