from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.errors import ResourceNotFoundError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext, WorkspaceAccessVerifier
    from hivememory.core.models import IdentityScope
    from hivememory.patchouli.runtime.bus import PatchouliBus


class MemoryTaskManagementService:
    """Patchouli 面向公开记忆任务 API 的应用服务。

    任务观察/取消的 operation 授权（``task.observe`` / ``management.task``）
    已上移到 workspace 能力层（A1 访问边界返工第 4.3 节）；本层只经
    ``verify_context`` 校验 access 有效性，缺少 access 一律拒绝。资源归属
    检查在取得必要投影后执行：任务必须携带归属投影且属于 access 上下文的
    Workspace（Workspace hard boundary），缺失归属或跨 Workspace 统一按
    not found 拒绝，不泄漏其他 Workspace 的任务存在性。
    """

    def __init__(
        self,
        *,
        bus: PatchouliBus,
        access_guard: WorkspaceAccessVerifier,
    ) -> None:
        # Public use-case 层只通过 local bus 访问任务控制面，避免直接持有 controller。
        self._bus = bus
        self._access_guard = access_guard

    async def list_memory_tasks(
        self,
        *,
        access: WorkspaceAccessContext,
    ) -> list[MemoryGenerationTask]:
        scope = self._access_guard.verify_context(access)
        tasks = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_LIST)
        # 归属投影检查在取得任务列表后执行，按 Workspace hard boundary 过滤。
        return [
            task
            for task in tasks
            if task.identity_scope is not None
            and task.identity_scope.workspace_identity == scope.workspace_identity
        ]

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> MemoryGenerationTask | None:
        scope = self._access_guard.verify_context(access)
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        self._assert_task_in_scope(scope, task, task_id)
        return task

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> bool:
        # 取消授权（management.task）在能力层；归属校验在取得投影后执行：
        # 不能取消其他 Workspace 的任务，也不暴露其存在。
        scope = self._access_guard.verify_context(access)
        task = await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)
        self._assert_task_in_scope(scope, task, task_id)
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)

    # ---- 内部辅助 ----

    def _assert_task_in_scope(
        self,
        scope: IdentityScope,
        task: MemoryGenerationTask | None,
        task_id: str,
    ) -> None:
        """任务归属校验：跨 Workspace/无归属与不存在统一 not found，不泄漏存在性。

        ``scope`` 是 access 校验返回的可信坐标；本断言只做资源归属投影比较
        （Workspace hard boundary），不重复执行行为授权。
        """
        task_scope = getattr(task, "identity_scope", None) if task is not None else None
        if (
            task is None
            or task_scope is None
            or task_scope.workspace_identity != scope.workspace_identity
        ):
            raise ResourceNotFoundError(details={"task_id": task_id})


__all__ = ["MemoryTaskManagementService"]
