"""MemoryTaskManagementService 归属校验的单元测试。

被测对象（A1 访问边界返工第 4.5/4.6 节）：
- 本层是授权点以下的资源 owner：``task.observe`` / ``management.task`` 的
  行为授权已上移 workspace 能力层，公开方法不接收 ``access`` 参数；
- 任务必须携带归属投影且属于 scope 的 Workspace：跨 Workspace 与不存在
  统一按 not found 拒绝，不泄漏其他 Workspace 的任务存在性；
- 列表按 scope 的 Workspace 过滤。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import IdentityScope
from hivememory.patchouli.application import MemoryTaskManagementService
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class FakeTaskBus:
    """以内存字典承载任务的路由假总线（控制器为边界外协作者）。"""

    def __init__(self, tasks):
        self._tasks = {task.task_id: task for task in tasks}
        self.cancelled: list[str] = []
        self.requested: list[str] = []

    async def request(self, route, *args, **kwargs):
        self.requested.append(route)
        if route == "memory_task.get":
            return self._tasks.get(args[0])
        if route == "memory_task.cancel":
            self.cancelled.append(args[0])
            return args[0] in self._tasks
        if route == "memory_task.list":
            return list(self._tasks.values())
        raise AssertionError(f"unexpected route: {route}")


def _task(task_id="active:i1", workspace=MAIN):
    scope = make_identity_scope(
        user_id=workspace.owner_user_id,
        agent_id="a1",
        workspace_id=workspace.workspace_id,
    )
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id="topic_1",
        label="topic_1",
        source=MemoryGenerationSource.WRITE,
        pending_alias="draft_x_0001",
        status=MemoryGenerationTaskStatus.PENDING,
        belong_to=workspace,
        from_actor=scope.actor_identity,
    )


def _run(coro):
    return asyncio.run(coro)


def _scope(workspace=MAIN) -> IdentityScope:
    return make_identity_scope(
        user_id=workspace.owner_user_id,
        agent_id="a1",
        workspace_id=workspace.workspace_id,
    )


def test_get_memory_task_returns_task_in_own_workspace():
    """归属投影与 scope 同 Workspace：返回任务只读快照。"""
    task = _task()
    service = MemoryTaskManagementService(bus=FakeTaskBus([task]))

    result = _run(service.get_memory_task(task.task_id, identity_scope=_scope()))

    assert result is task
    assert result.belong_to == MAIN


def test_get_memory_task_hides_cross_workspace_and_missing_as_not_found():
    """跨 Workspace 与不存在统一 not found，不泄漏存在性。"""
    foreign = _task(workspace=OTHER)
    service = MemoryTaskManagementService(bus=FakeTaskBus([foreign]))

    scope = _scope()
    with pytest.raises(ResourceNotFoundError, match="workspace.resource.not_found") as excinfo:
        _run(service.get_memory_task(foreign.task_id, identity_scope=scope))
    assert excinfo.value.details == {"task_id": foreign.task_id}

    with pytest.raises(ResourceNotFoundError, match="workspace.resource.not_found") as excinfo:
        _run(service.get_memory_task("active:ghost", identity_scope=scope))
    assert excinfo.value.details == {"task_id": "active:ghost"}


def test_cancel_memory_task_in_own_workspace_cancels():
    """归属一致的任务可取消：取消请求到达控制器。"""
    task = _task()
    bus = FakeTaskBus([task])

    result = _run(
        MemoryTaskManagementService(bus=bus).cancel_memory_task(
            task.task_id, identity_scope=_scope()
        )
    )

    assert result is True
    assert bus.cancelled == [task.task_id]


def test_cancel_memory_task_rejects_cross_workspace_as_not_found():
    """另一 Workspace 的 scope 不能取消跨域任务：not found 且无取消副作用。"""
    task = _task()
    bus = FakeTaskBus([task])
    foreign_scope = _scope(OTHER)

    with pytest.raises(ResourceNotFoundError, match="workspace.resource.not_found"):
        _run(
            MemoryTaskManagementService(bus=bus).cancel_memory_task(
                task.task_id, identity_scope=foreign_scope
            )
        )
    assert bus.cancelled == []


def test_list_memory_tasks_filters_to_own_workspace():
    """列表按 scope 的 Workspace 过滤，不泄漏其他归属的任务。"""
    own = _task()
    foreign = _task(task_id="active:i2", workspace=OTHER)
    bus = FakeTaskBus([own, foreign])

    filtered = _run(MemoryTaskManagementService(bus=bus).list_memory_tasks(identity_scope=_scope()))
    assert [t.task_id for t in filtered] == [own.task_id]

    # 同一任务面对另一 Workspace 的 scope 时角色互换：过滤器只由 scope 决定
    foreign_view = _run(
        MemoryTaskManagementService(bus=bus).list_memory_tasks(identity_scope=_scope(OTHER))
    )
    assert [t.task_id for t in foreign_view] == [foreign.task_id]


def test_constructor_rejects_access_guard_and_methods_reject_access_parameter():
    """授权点参数不再出现在本层签名：构造与公开方法均不接受 access。"""
    with pytest.raises(TypeError, match="access_guard"):
        MemoryTaskManagementService(bus=FakeTaskBus([]), access_guard=object())

    service = MemoryTaskManagementService(bus=FakeTaskBus([_task()]))
    with pytest.raises(TypeError, match="access"):
        _run(service.get_memory_task("active:i1", identity_scope=_scope(), access=object()))
    with pytest.raises(TypeError, match="access"):
        _run(service.cancel_memory_task("active:i1", identity_scope=_scope(), access=object()))


@pytest.mark.parametrize(
    "invoke",
    [
        lambda svc: svc.list_memory_tasks(),
        lambda svc: svc.get_memory_task("active:i1"),
        lambda svc: svc.cancel_memory_task("active:i1"),
    ],
    ids=["list_memory_tasks", "get_memory_task", "cancel_memory_task"],
)
def test_missing_identity_scope_rejected_as_required_argument(invoke):
    """identity_scope 缺失由必填签名拒绝，且不触达任务控制面。"""
    bus = FakeTaskBus([])
    service = MemoryTaskManagementService(bus=bus)

    with pytest.raises(TypeError, match="identity_scope"):
        _run(invoke(service))
    assert bus.requested == []
