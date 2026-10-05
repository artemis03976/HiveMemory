from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import IdentityScope, WorkspaceIdentity, require_identity_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class MemoryTaskManagementService:
    """Patchouli 面向公开记忆任务 API 的应用服务。

    任务观察/取消的 operation 授权（``task.observe`` / ``management.task``）
    已上移到 workspace 能力层（A1 访问边界返工第 4.5 节）；本层是授权点
    以下的资源 owner，只接收授权点组装的 ``IdentityScope``，不接收访问
    context。资源归属检查在取得必要投影后执行：任务必须携带归属投影且
    属于 scope 的 Workspace（Workspace hard boundary），跨
    Workspace 统一按 not found 拒绝，不泄漏其他 Workspace 的任务存在性。
    """

    def __init__(self, *, bus: PatchouliBus) -> None:
        # Public use-case 层只通过 local bus 访问任务控制面，避免直接持有 controller。
        self._bus = bus

    async def list_memory_tasks(
        self,
        *,
        identity_scope: IdentityScope,
    ) -> list[MemoryGenerationTask]:
        scope = require_identity_scope(identity_scope)
        tasks = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_LIST)
        # 归属投影检查在取得任务列表后执行，按 Workspace hard boundary 过滤。
        return [task for task in tasks if task.belong_to == scope.workspace_identity]

    async def get_memory_task(
        self,
        task_id: str,
        *,
        identity_scope: IdentityScope,
    ) -> MemoryGenerationTask | None:
        scope = require_identity_scope(identity_scope)
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        self._assert_task_in_workspace(scope.workspace_identity, task, task_id)
        return task

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        identity_scope: IdentityScope,
    ) -> bool:
        # 取消授权（management.task）在能力层；归属校验在取得投影后执行：
        # 不能取消其他 Workspace 的任务，也不暴露其存在。
        scope = require_identity_scope(identity_scope)
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        self._assert_task_in_workspace(scope.workspace_identity, task, task_id)
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)

    # ---- 内部辅助 ----

    def _assert_task_in_workspace(
        self,
        belong_to: WorkspaceIdentity,
        task: MemoryGenerationTask | None,
        task_id: str,
    ) -> None:
        """任务归属校验：跨 Workspace 与不存在统一 not found，不泄漏存在性。

        ``belong_to`` 在公开边界从授权 scope 拆出；本断言只做资源归属投影
        比较（Workspace hard boundary），不重复执行行为授权。
        """
        if task is None or task.belong_to != belong_to:
            raise ResourceNotFoundError(details={"task_id": task_id})


__all__ = ["MemoryTaskManagementService"]
