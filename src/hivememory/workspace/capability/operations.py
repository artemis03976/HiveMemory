"""workspace 操作入口：校验线程 context，直接分派请求。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from hivememory.core.access import WorkspaceAccessContext
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
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


class WorkspaceOperationEntry:
    """入口不拆业务参数，身份与进程关联均经授权者从线程 context 取得。"""

    def __init__(
        self,
        memory: ActorMemoryService,
        *,
        agent: ActorAgentService,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._authorizer = operation_authorizer
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
        """线程有效才分派；写入的登记前复查与派发由 actor API 负责。"""
        self._authorizer.validate_thread_context(access)
        handler = self._handlers.get(type(request))
        if handler is None:
            raise TypeError(f"Unsupported operation request: {type(request).__name__}")
        return cast(R, await handler(request, access=access))


__all__ = ["WorkspaceOperationEntry"]
