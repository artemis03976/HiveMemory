"""workspace 操作入口：经线程表判定线程存活，直接分派请求。"""

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
from hivememory.workspace.thread_ports import ExecutionThreadTable


class WorkspaceOperationEntry:
    """入口不拆业务参数；身份由 actor API 经授权者从线程 context 取得。"""

    def __init__(
        self,
        memory: ActorMemoryService,
        *,
        agent: ActorAgentService,
        threads: ExecutionThreadTable,
    ) -> None:
        self._threads = threads
        # 泛型结果由公开契约约束；异构表只在此擦除类型，直接指向 actor 方法。
        self._handlers: dict[type[OperationRequest[Any]], Callable[..., Awaitable[Any]]] = {
            SubmitWriteIntentRequest: memory.submit_write_intent,
            SubmitUpdateIntentRequest: memory.submit_update_intent,
            ResolveReferencesRequest: memory.resolve_references,
            ResolveRunTargetRequest: memory.resolve_run_target,
            RetrieveRequest: memory.retrieve,
            DeriveCallThreadRequest: agent.derive_call_thread,
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


__all__ = ["WorkspaceOperationEntry"]
