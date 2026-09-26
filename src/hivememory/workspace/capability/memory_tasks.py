"""记忆生成任务能力：任务查询/取消薄委托（A2 §1.2，自 ``system/application`` 迁入）。

任务快照类型取自公共契约 ``patchouli.contracts.memory_tasks``，不依赖控制面
实现；operation 检查（``task.observe`` / ``management.task``）仍由 Patchouli
application 执行。

TODO(A5/A6)：能力层依赖 ``system.*`` / ``patchouli.contracts`` 属过渡期分层导入
白名单（A2 §8 D-2），能力层与 system 的依赖方向届时重新整理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTask
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.workspace import WorkspaceAccessContext


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

    def __init__(self, global_bus: GlobalSystemBus) -> None:
        self._global_bus = global_bus

    async def list_memory_tasks(
        self,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> list[MemoryGenerationTask]:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST,
            access=access,
        )

    async def get_memory_task(
        self,
        task_id: str,
        *,
        access: WorkspaceAccessContext | None = None,
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
        access: WorkspaceAccessContext | None = None,
    ) -> bool:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
            access=access,
        )


__all__ = ["MemoryTaskApplicationService"]
