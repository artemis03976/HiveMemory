"""workspace 操作入口：经线程表判定线程存活，直接分派记忆请求，并提交线程的生命周期动作。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from hivememory.core.access import WorkspaceAccessContext
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.contracts.operations import (
    DeriveCallThreadRequest,
    OperationRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.contracts.threads import ExecutionThread
from hivememory.workspace.thread_ports import ExecutionThreadTable


class WorkspaceOperationEntry:
    """入口不拆业务参数；身份由 actor API 经授权者从线程 context 取得。

    被调用方的身份必须来自 workspace 自己的解析：CALL 的图纸解析与子线程
    派生在 :meth:`derive_child_thread` 的一次提交内依次完成，CPU 不能交回
    一份图纸来自报子线程的身份。
    """

    def __init__(
        self,
        memory: ActorMemoryService,
        *,
        agent: ActorAgentService,
        threads: ExecutionThreadTable,
    ) -> None:
        self._threads = threads
        self._agent = agent
        # 泛型结果由公开契约约束；异构表只在此擦除类型，直接指向 actor 方法。
        self._handlers: dict[type[OperationRequest[Any]], Callable[..., Awaitable[Any]]] = {
            SubmitWriteIntentRequest: memory.submit_write_intent,
            SubmitUpdateIntentRequest: memory.submit_update_intent,
            ResolveReferencesRequest: memory.resolve_references,
            ResolveRunTargetRequest: memory.resolve_run_target,
            RetrieveRequest: memory.retrieve,
        }

    async def execute[R](
        self, request: OperationRequest[R], *, access: WorkspaceAccessContext
    ) -> R:
        """线程存活才分派；写入的登记前复查与派发由 actor API 负责。

        全部 actor 操作在此共用线程表的同一个存活判定：线程已结束或进程已
        收到停止时以 ``ThreadEndedError`` 拒绝，拒绝时没有任何副作用。
        """
        self._threads.resolve(access)
        handler = self._handlers.get(type(request))
        if handler is None:
            raise TypeError(f"Unsupported operation request: {type(request).__name__}")
        return cast(R, await handler(request, access=access))

    async def derive_child_thread(
        self, request: DeriveCallThreadRequest, *, access: WorkspaceAccessContext
    ) -> ExecutionThread:
        """CALL 派生：确认调用方存活，授权 ``agent.call`` 并解析图纸，再交给线程表派生。

        图纸冷读之后由线程表重新判定调用方存活并确认它是主线程，再以被调用方
        准入后派生、读取操作快照并登记；任一步拒绝都不留下凭据或登记。
        """
        self._threads.resolve(access)
        profile = await self._agent.resolve_call_target(request, access=access)
        return self._threads.derive_child(parent=access, profile=profile)

    def end_thread(self, *, access: WorkspaceAccessContext) -> None:
        """同步结束子线程；已结束的线程是空操作，主线程以接线缺陷拒绝。"""
        self._threads.end_child(access)


__all__ = ["WorkspaceOperationEntry"]
