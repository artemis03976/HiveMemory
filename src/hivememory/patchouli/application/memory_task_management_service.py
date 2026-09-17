from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.errors import ResourceNotFoundError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class MemoryTaskManagementService:
    """Patchouli 面向公开记忆任务 API 的应用服务。

    任务观察/等待/取消的授权（父计划 5.6.4/5.7.1，WRX-1 冻结）：

    - 提供 ``access`` 时：get/wait/list 绑定 ``task.observe``，cancel 绑定
      ``management.task``（观察不授予取消）；任务必须携带归属投影且与
      access 上下文一致，缺失归属或跨 scope 统一按 not found 拒绝，
      不泄漏其他 Workspace 的任务存在性；
    - 未提供 ``access`` 的旧调用方（Patchouli 内部 finalize/wait 链路）为
      迁移期受信适配，保持既有行为；消费者切换在 WRX-4/5 完成后收紧。
    """

    def __init__(self, *, bus: PatchouliBus) -> None:
        # Public use-case 层只通过 local bus 访问任务控制面，避免直接持有 controller。
        self._bus = bus

    async def list_memory_tasks(
        self,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> list[MemoryGenerationTask]:
        tasks = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_LIST)
        if access is None:
            return tasks
        context = require_access_context(access, operation=WorkspaceOperation.TASK_OBSERVE)
        return [
            task
            for task in tasks
            if task.identity_scope is not None and task.identity_scope == context.identity_scope
        ]

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryGenerationTask | None:
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        if access is not None:
            self._assert_observable(access, task, task_id)
        return task

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> bool:
        if access is not None:
            # 取消绑定 management.task（task.observe 不授予取消）；归属校验
            # 先行：不能取消其他 Workspace 的任务，也不暴露其存在。
            context = require_access_context(access, operation=WorkspaceOperation.MANAGEMENT_TASK)
            task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
            self._assert_scope_matches(context, task, task_id)
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)

    async def wait_memory_task(
        self,
        task_id: str,
        timeout: float | None = None,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryGenerationTask | None:
        if access is not None:
            task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
            self._assert_observable(access, task, task_id)
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

    def _assert_observable(
        self,
        access: WorkspaceAccessContext,
        task: MemoryGenerationTask | None,
        task_id: str,
    ) -> None:
        """task.observe 的归属校验：跨 scope/无归属与不存在统一 not found。"""
        context = require_access_context(access, operation=WorkspaceOperation.TASK_OBSERVE)
        self._assert_scope_matches(context, task, task_id)

    @staticmethod
    def _assert_scope_matches(
        context: WorkspaceAccessContext,
        task: MemoryGenerationTask | None,
        task_id: str,
    ) -> None:
        task_scope = getattr(task, "identity_scope", None) if task is not None else None
        if task is None or task_scope != context.identity_scope:
            raise ResourceNotFoundError(details={"task_id": task_id})


__all__ = ["MemoryTaskManagementService"]
