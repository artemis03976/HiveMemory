"""Patchouli 记忆生成任务的应用层门面。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.system.access import WorkspaceAccessContext


class MemoryTaskApplicationService:
    """
    Top-level API facade for memory task query and cancellation.

    MemoryGenerationTask 的生命周期仍由 Patchouli 拥有；顶层 service 只通过
    GlobalSystemBus 请求 Patchouli 公开 API，保持 system/application service 与
    其它子系统能力访问方式一致。

    访问上下文约定（A1 计划第 1.1/3.3 节）：``access`` 为统一认证网关
    签发的可信 context，原样透传——观察（list/get/wait）绑定
    ``task.observe``，取消绑定 ``management.task``，行为检查先于后端
    读取，任务归属校验在 Patchouli application 落实。
    """

    def __init__(self, global_bus: "GlobalSystemBus") -> None:
        self._global_bus = global_bus

    async def list_memory_tasks(
        self,
        *,
        access: "WorkspaceAccessContext | None" = None,
    ) -> list[MemoryGenerationTask]:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST,
            access=access,
        )

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: "WorkspaceAccessContext | None" = None,
    ) -> MemoryGenerationTask | None:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            task_id,
            access=access,
        )

    async def cancel_memory_task(
        self,
        task_id: str,
        *,
        access: "WorkspaceAccessContext | None" = None,
    ) -> bool:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
            access=access,
        )


__all__ = ["MemoryTaskApplicationService"]
