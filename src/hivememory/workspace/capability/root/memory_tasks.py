"""root 记忆生成任务 API：授权后查询或取消 Patchouli 拥有的任务。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootMemoryTaskService:
    """root 任务查询与取消入口，任务生命周期与归属校验由 Patchouli 拥有。"""

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
        """列出本 Workspace 的记忆生成任务（root 授权）。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
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
    ) -> MemoryGenerationTask:
        """读取单个记忆生成任务（root 授权）。

        不存在与越域由资源 owner 统一以 ``ResourceNotFoundError`` 拒绝，不返回 ``None``。
        """
        scope = self._authorizer.authorize_root(access, target_workspace)
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
        """取消记忆生成任务（root 管理操作）。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
            identity_scope=scope,
        )


__all__ = ["RootMemoryTaskService"]
