"""workspace 操作入口：校验线程 context，直接分派请求并补偿迟到意图。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.errors import ThreadContextRevokedError
from hivememory.core.models.pending import PendingAtom
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.contracts.operations import (
    CancelIntentsRequest,
    OperationRequest,
    ResolveCallTargetRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.intents.registry import WriteIntentRegistry

_INTENT_SUBMISSIONS = (SubmitWriteIntentRequest, SubmitUpdateIntentRequest)


class WorkspaceOperationEntry:
    """入口不拆业务参数，身份与进程关联均经授权者从线程 context 取得。"""

    def __init__(
        self,
        memory: ActorMemoryService,
        *,
        agent: ActorAgentService,
        operation_authorizer: WorkspaceOperationAuthorizer,
        intent_registry: WriteIntentRegistry,
    ) -> None:
        self._authorizer = operation_authorizer
        self._intents = intent_registry
        # 泛型结果由公开契约约束；异构表只在此擦除类型，直接指向 actor 方法。
        self._handlers: dict[type[OperationRequest[Any]], Callable[..., Awaitable[Any]]] = {
            SubmitWriteIntentRequest: memory.submit_write_intent,
            SubmitUpdateIntentRequest: memory.submit_update_intent,
            ResolveReferencesRequest: memory.resolve_references,
            ResolveRunTargetRequest: memory.resolve_run_target,
            RetrieveRequest: memory.retrieve,
            ResolveCallTargetRequest: agent.resolve_call_target,
        }

    async def execute[R](
        self, request: OperationRequest[R], *, access: WorkspaceAccessContext
    ) -> R:
        """线程有效才分派，意图提交返回后在同步段复查并补偿撤销期间的登记。

        只读请求通过授权后可自然完成。WRITE/UPDATE 在 await 期间可能跨越
        CPU 结束或进程关闭，因此先保留进程关联；线程撤销后仍可据此同步
        撤回本次登记，不重新授权、不取消调用方，也不返回失效的 ACK。
        """
        self._authorizer.validate_thread_context(access)
        if type(request) is CancelIntentsRequest:
            # 撤回是内部补偿，不检查 actor operation，只作用于本进程的 PENDING。
            return cast(
                R,
                self._intents.cancel_aliases(
                    list(request.aliases),
                    process_id=self._authorizer.get_thread_process_id(access),
                ),
            )
        handler = self._handlers.get(type(request))
        if handler is None:
            raise TypeError(f"Unsupported operation request: {type(request).__name__}")
        process_id = (
            self._authorizer.get_thread_process_id(access)
            if isinstance(request, _INTENT_SUBMISSIONS)
            else None
        )
        result = await handler(request, access=access)
        if process_id is not None:
            try:
                self._authorizer.validate_thread_context(access)
            except ThreadContextRevokedError:
                # 能力返回至补偿没有挂起点，不能留下游离意图或误撤其他进程。
                pending = cast(PendingAtom, result)
                self._intents.cancel_aliases([pending.pending_alias], process_id=process_id)
                raise
        return cast(R, result)


__all__ = ["WorkspaceOperationEntry"]
