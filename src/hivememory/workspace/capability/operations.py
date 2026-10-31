"""workspace 操作入口：actor API 唯一的调用方。

执行线程的提交经线程表判定线程存活后分派，并提交线程的生命周期动作；
不建进程的外部入口以请求级 context 提交检索、读取、写入与更新。
"""

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

# 请求级提交开放检索、读取、写入与更新：RUN 的执行发生在执行单元的环境中，
# CALL 只属于执行线程。请求级 context 没有线程，写入与更新登记后以调用方给出的
# 派发话题派发，检索与读取不涉及话题。
_REQUEST_READS: frozenset[type[OperationRequest[Any]]] = frozenset(
    {RetrieveRequest, ResolveReferencesRequest}
)
_REQUEST_INTENTS: frozenset[type[OperationRequest[Any]]] = frozenset(
    {SubmitWriteIntentRequest, SubmitUpdateIntentRequest}
)


class WorkspaceOperationEntry:
    """入口不拆业务参数；身份由 actor API 经授权者从访问 context 取得。

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

        线程提交的全部 actor 操作在此共用线程表的同一个存活判定：线程已结束
        或进程已收到停止时以 ``ThreadEndedError`` 拒绝，进程 context 与请求级
        context 以 ``ScopeRequiredError`` 拒绝，拒绝时没有任何副作用。
        """
        self._threads.resolve(access)
        handler = self._handlers.get(type(request))
        if handler is None:
            raise TypeError(f"Unsupported operation request: {type(request).__name__}")
        return cast(R, await handler(request, access=access))

    async def execute_request[R](
        self,
        request: OperationRequest[R],
        *,
        access: WorkspaceAccessContext,
        dispatch_topic_id: str | None = None,
    ) -> R:
        """以请求级 context 提交检索、读取、写入或更新，与线程提交共用同一分派表。

        写入与更新必须给出派发话题，即调用方为这一轮落位得到的话题；检索与
        读取不得给出。其余请求类型、或派发话题与请求类型不符时，在分派前以
        ``TypeError`` 拒绝，没有副作用：外部入口只按这两条规则构造请求，不符
        即是接线缺陷。请求级 context 只在本次调用内有效，不另做存活判定，撤销
        由授权检查。
        """
        request_type = type(request)
        if request_type in _REQUEST_READS:
            if dispatch_topic_id is not None:
                raise TypeError(f"{request_type.__name__} does not take a dispatch topic")
            return cast(R, await self._handlers[request_type](request, access=access))
        if request_type in _REQUEST_INTENTS:
            if dispatch_topic_id is None:
                raise TypeError(f"{request_type.__name__} requires a dispatch topic")
            return cast(
                R,
                await self._handlers[request_type](
                    request, access=access, dispatch_topic_id=dispatch_topic_id
                ),
            )
        raise TypeError(f"Unsupported request-level operation: {request_type.__name__}")

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
