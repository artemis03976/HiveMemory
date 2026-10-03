"""记忆生成任务能力：任务查询/取消薄委托（A2 §1.2，自 ``system/application`` 迁入）。

任务快照类型取自公共契约 ``patchouli.contracts.memory_tasks``，不依赖控制面
实现；operation 授权（观察 ``task.observe``、取消 ``management.task``）在本
层、路由调用前执行（A1 访问边界返工第 4.3 节）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class MemoryTaskApplicationService:
    """
    Top-level API facade for memory task query and cancellation.

    MemoryGenerationTask 的生命周期仍由 Patchouli 拥有；顶层 service 只通过
    GlobalSystemBus 请求 Patchouli 公开 API，保持 system/application service 与
    其它子系统能力访问方式一致。

    访问上下文约定（A1 访问边界返工第 4.3 节）：``access`` 为统一认证网关
    签发的可信 context，行为检查先于后端读取——观察（list/get）绑定
    ``task.observe``，取消绑定 ``management.task``；任务归属校验在 Patchouli
    application 落实。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        self._global_bus = global_bus
        self._access_guard = access_guard

    async def list_memory_tasks(
        self,
        *,
        access: WorkspaceAccessContext,
    ) -> list[MemoryGenerationTask]:
        """列出本 Workspace 的记忆生成任务（``task.observe``）。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.TASK_OBSERVE)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST,
            access=access,
        )

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> MemoryGenerationTask | None:
        """读取单个记忆生成任务（``task.observe``）。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.TASK_OBSERVE)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            task_id,
            access=access,
        )

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> bool:
        """取消记忆生成任务（``management.task``；观察不授予取消）。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.MANAGEMENT_TASK)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
            access=access,
        )


__all__ = ["MemoryTaskApplicationService"]
