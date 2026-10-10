"""单个任务进程的操作通道，绑定主线程访问凭据与注册时的目标。

执行者只取得 contracts 中的端口视图；访问 context 始终留在进程内部。
关闭是同步操作，先于 CPU 输出流关闭和 prepare 补偿中的任何 await。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models import WorkspaceIdentity
from hivememory.core.models.pending import PendingAtom, WriteFocus
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.contracts.operations import ProcessOperationsClosedError


class ProcessOperationChannel:
    """绑定一个进程的操作通道，每次操作仍经能力层逐次授权。"""

    def __init__(
        self,
        memory: MemoryApplicationService,
        *,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
        process_id: str,
    ) -> None:
        self._memory = memory
        self._access: WorkspaceAccessContext | None = access
        self._target_workspace = target_workspace
        self._process_id = process_id
        self._active_calls: set[asyncio.Task] = set()

    def close(self) -> None:
        """同步使通道失效，并释放通道持有的访问 context。"""
        self._access = None
        # UPDATE 的基础原子可能在冷读中。关闭时同步请求取消在途操作，
        # 防止它在关闭流程已取消 PENDING 之后才登记出新的意图。
        for task in tuple(self._active_calls):
            task.cancel()

    def _require_access(self) -> WorkspaceAccessContext:
        if self._access is None:
            raise ProcessOperationsClosedError("Task process operations are closed")
        return self._access

    async def _invoke[T](self, operation: Callable[[WorkspaceAccessContext], Awaitable[T]]) -> T:
        """登记在途调用；关闭只取消调用方任务，不创建额外后台任务。"""
        access = self._require_access()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Process operations require an asyncio task")
        self._active_calls.add(task)
        try:
            return await operation(access)
        finally:
            self._active_calls.discard(task)

    async def submit_write_intent(self, focus: WriteFocus) -> PendingAtom:
        """WRITE 以绑定身份登记，进程 ID 仅用于后续认领。"""
        return await self._invoke(
            lambda access: self._memory.submit_write_intent(
                focus=focus,
                process_id=self._process_id,
                target_workspace=self._target_workspace,
                access=access,
            )
        )

    async def submit_update_intent(
        self, base_alias: str, instruction: str, content: str | None = None
    ) -> PendingAtom:
        """UPDATE 的基础原子解析与登记均由能力层负责。"""
        return await self._invoke(
            lambda access: self._memory.submit_update_intent(
                base_alias,
                instruction,
                content,
                process_id=self._process_id,
                target_workspace=self._target_workspace,
                access=access,
            )
        )

    async def cancel_intents(self, aliases: list[str]) -> list[str]:
        """只撤回本进程仍为 PENDING 的意图，进程 ID 由通道绑定。"""
        return await self._invoke(
            lambda access: self._memory.cancel_intents(
                aliases,
                process_id=self._process_id,
                target_workspace=self._target_workspace,
                access=access,
            )
        )

    async def resolve_references(self, aliases: list[str]) -> list[ReferenceResolution]:
        """以绑定的目标读取引用，不接受执行者提交的身份坐标。"""
        return await self._invoke(
            lambda access: self._memory.resolve_references(
                aliases,
                target_workspace=self._target_workspace,
                access=access,
            )
        )


__all__ = ["ProcessOperationChannel"]
