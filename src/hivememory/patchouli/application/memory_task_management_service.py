from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.errors import ResourceNotFoundError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.core.models import IdentityScope
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class MemoryTaskManagementService:
    """Patchouli 面向公开记忆任务 API 的应用服务。

    任务观察/等待/取消的授权（A1 计划第 3.4/4.1 节）：行为检查一律先于
    后端读取，资源归属检查在取得必要投影后执行——

    - 提供 ``access`` 时：get/wait/list 绑定 ``task.observe``，cancel 绑定
      ``management.task``（观察不授予取消）；任务必须携带归属投影且与
      access 上下文一致，缺失归属或跨 scope 统一按 not found 拒绝，
      不泄漏其他 Workspace 的任务存在性；
    - 未提供 ``access`` 的旧调用方（Patchouli 内部 finalize/wait 链路）为
      A1 第 6 节兼容清单内的迁移期受信适配，保持既有行为；A6 完成消费
      者切换后收紧。

    ``wait_memory_tasks``/``wait_all_memory_tasks`` 是内部等待用例（未
    挂载公开路由），不进入 Actor 观察授权面；公开 ``wait`` 如需暴露，
    须先按 A1 第 3.4 节冻结等待上限、超时与 shutdown 结束语义。
    """

    def __init__(
        self,
        *,
        bus: PatchouliBus,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        # Public use-case 层只通过 local bus 访问任务控制面，避免直接持有 controller。
        self._bus = bus
        self._access_guard = access_guard

    async def list_memory_tasks(
        self,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> list[MemoryGenerationTask]:
        scope = (
            None
            if access is None
            else self._access_guard.authorize_operation(access, WorkspaceOperation.TASK_OBSERVE)
        )
        tasks = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_LIST)
        if scope is None:
            return tasks
        # 归属投影检查在取得任务列表后执行。
        return [
            task
            for task in tasks
            if task.identity_scope is not None and task.identity_scope == scope
        ]

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryGenerationTask | None:
        if access is not None:
            scope = self._access_guard.authorize_operation(access, WorkspaceOperation.TASK_OBSERVE)
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        if access is not None:
            self._assert_scope_matches(scope, task, task_id)
        return task

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> bool:
        if access is not None:
            # 取消绑定 management.task（task.observe 不授予取消）；行为检查
            # 先行，归属校验在取得投影后执行：不能取消其他 Workspace 的
            # 任务，也不暴露其存在。
            scope = self._access_guard.authorize_operation(
                access, WorkspaceOperation.MANAGEMENT_TASK
            )
            task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
            self._assert_scope_matches(scope, task, task_id)
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)

    async def wait_memory_task(
        self,
        task_id: str,
        timeout: float | None = None,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryGenerationTask | None:
        if access is not None:
            # 公开等待与观察同一 operation；归属校验先于等待副作用。
            scope = self._access_guard.authorize_operation(access, WorkspaceOperation.TASK_OBSERVE)
            task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
            self._assert_scope_matches(scope, task, task_id)
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_WAIT,
            task_id,
            timeout,
        )

    async def wait_memory_tasks(
        self,
        task_ids: list[str],
        timeout: float | None = None,
    ) -> list[MemoryGenerationTask | None]:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_WAIT_MANY,
            task_ids,
            timeout,
        )

    async def wait_all_memory_tasks(
        self,
        timeout: float | None = None,
    ) -> list[MemoryGenerationTask]:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_WAIT_ALL,
            timeout,
        )

    # ---- 内部辅助 ----

    def _assert_scope_matches(
        self,
        scope: IdentityScope,
        task: MemoryGenerationTask | None,
        task_id: str,
    ) -> None:
        """任务归属校验：跨 scope/无归属与不存在统一 not found，不泄漏存在性。

        ``scope`` 是行为检查返回的可信坐标；本断言只做资源归属投影比较，
        不重复执行行为授权。
        """
        task_scope = getattr(task, "identity_scope", None) if task is not None else None
        if task is None or task_scope != scope:
            raise ResourceNotFoundError(details={"task_id": task_id})


__all__ = ["MemoryTaskManagementService"]
