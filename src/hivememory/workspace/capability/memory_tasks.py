"""记忆生成任务能力：任务查询/取消薄委托（A2 §1.2，自 ``system/application`` 迁入）。

任务快照类型取自公共契约 ``patchouli.contracts.memory_tasks``，不依赖控制面
实现；操作授权（观察 ``task.observe``、取消 ``management.task``）在本层、
路由调用前执行（A1 访问边界返工第 4.5 节），任务归属校验在 Patchouli
application 落实。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import IdentityScope, WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class MemoryTaskApplicationService:
    """
    Top-level API facade for memory task query and cancellation.

    MemoryGenerationTask 的生命周期仍由 Patchouli 拥有；顶层 service 只通过
    GlobalSystemBus 请求 Patchouli 公开 API，保持应用服务与其它子系统能力
    访问方式一致。

    访问上下文约定（A1 访问边界返工第 4.5 节）：本层是授权点——方法只
    接收访问 context 与目标 workspace；行为检查先于后端读取——观察
    （list/get）绑定 ``task.observe``，取消绑定 ``management.task``；
    Patchouli 只接收操作授权者返回的可信 scope。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer

    async def list_memory_tasks(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> list[MemoryGenerationTask]:
        """列出本 Workspace 的记忆生成任务（``task.observe``）。"""
        scope = self._authorize(access, WorkspaceOperation.TASK_OBSERVE, target_workspace)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST,
            identity_scope=scope,
        )

    async def get_memory_task(
        self,
        task_id: str,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> MemoryGenerationTask | None:
        """读取单个记忆生成任务（``task.observe``）。"""
        scope = self._authorize(access, WorkspaceOperation.TASK_OBSERVE, target_workspace)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            task_id,
            identity_scope=scope,
        )

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> bool:
        """取消记忆生成任务（``management.task``；观察不授予取消）。"""
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_TASK, target_workspace)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
            identity_scope=scope,
        )

    def _authorize(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """在后端读取/取消前执行操作授权，返回组装的可信 scope。"""
        return self._authorizer.authorize_operation(access, operation, target_workspace)


__all__ = ["MemoryTaskApplicationService"]
