"""workspace 操作入口：actor API 唯一的调用方。

执行线程的提交以线程句柄经线程表判定线程存活后分派，并提交线程的生命周期
动作；不建进程的外部入口以入口认证得到的工作 actor 提交检索、读取、写入与
更新。每一次 actor 操作在结束时发一个 ``actor.operation`` 事件。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, cast

from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkingActor, WorkspaceOperation
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService, DispatchTopic
from hivememory.workspace.capability.operation_events import (
    ActorOperationEventEmitter,
    operation_of,
)
from hivememory.workspace.contracts.operations import (
    DeriveCallThreadRequest,
    OperationRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.contracts.threads import ExecutionThread, ThreadHandle
from hivememory.workspace.thread_ports import ExecutionThreadTable

# 写入与更新登记意图后要以派发话题派发：线程提交向线程表取得进程的派发话题，
# 请求级提交调用入口给出的回调确定话题；其余请求不涉及话题。
_INTENT_REQUESTS: frozenset[type[OperationRequest[Any]]] = frozenset(
    {SubmitWriteIntentRequest, SubmitUpdateIntentRequest}
)
# 请求级提交开放检索、读取、写入与更新：RUN 的执行发生在执行单元的环境中，
# CALL 只属于执行线程。
_REQUEST_READS: frozenset[type[OperationRequest[Any]]] = frozenset(
    {RetrieveRequest, ResolveReferencesRequest}
)


class WorkspaceOperationEntry:
    """入口不拆业务参数；actor API 以入口交出的工作 actor 逐次授权。

    线程提交的工作 actor 取自线程表的解析结果，CPU 一侧只持有不透明的线程
    句柄，取不到工作 actor。被调用方的身份必须来自 workspace 自己的解析：
    CALL 的图纸解析与子线程派生在 :meth:`derive_child_thread` 的一次提交内
    依次完成，CPU 不能交回一份图纸来自报子线程的身份。
    """

    def __init__(
        self,
        memory: ActorMemoryService,
        *,
        agent: ActorAgentService,
        threads: ExecutionThreadTable,
        runtime_events: RuntimeEventPublisher | None = None,
    ) -> None:
        self._threads = threads
        self._agent = agent
        self._events = ActorOperationEventEmitter(runtime_events)
        # 泛型结果由公开契约约束；异构表只在此擦除类型，直接指向 actor 方法。
        self._handlers: dict[type[OperationRequest[Any]], Callable[..., Awaitable[Any]]] = {
            SubmitWriteIntentRequest: memory.submit_write_intent,
            SubmitUpdateIntentRequest: memory.submit_update_intent,
            ResolveReferencesRequest: memory.resolve_references,
            ResolveRunTargetRequest: memory.resolve_run_target,
            RetrieveRequest: memory.retrieve,
        }

    async def execute[R](self, request: OperationRequest[R], *, thread: ThreadHandle) -> R:
        """线程存活才分派；写入的登记前复查与派发由 actor API 负责。

        线程提交的全部 actor 操作在此共用线程表的同一个存活判定：线程已结束、
        进程已收到停止或已经关闭时以 ``ThreadEndedError`` 拒绝，句柄不是登记的
        那一个以 ``ScopeRequiredError`` 拒绝，拒绝时没有任何副作用。解析得到的
        工作 actor 交给 actor API 授权；写入与更新另带句柄，供登记前的同步段
        再解析一次线程。
        """
        resolved = self._threads.resolve(thread)
        actor = resolved.actor
        request_type = type(request)
        handler = self._handlers.get(request_type)
        if handler is None:
            raise TypeError(f"Unsupported operation request: {request_type.__name__}")
        if request_type in _INTENT_REQUESTS:
            run = partial(handler, request, actor=actor, thread=thread)
        else:
            run = partial(handler, request, actor=actor)
        result = await self._events.observe(
            operation_of(request),
            request,
            run,
            actor=actor,
            entry="thread",
            process_id=resolved.process_id,
        )
        return cast(R, result)

    async def execute_request[R](
        self,
        request: OperationRequest[R],
        *,
        actor: WorkingActor,
        dispatch_topic: DispatchTopic | None = None,
    ) -> R:
        """以入口认证得到的工作 actor 提交检索、读取、写入或更新，与线程提交共用同一分派表。

        写入与更新必须给出 ``dispatch_topic``：为这一轮确定派发话题的回调，actor
        API 在授权之后才调用它，被拒的写入因此不会确定话题；检索与读取不得
        给出。其余请求类型、或回调与请求类型不符时，在分派前以 ``TypeError``
        拒绝，没有副作用：外部入口只按这两条规则构造请求，不符即是接线缺陷。
        工作 actor 只是入口调用链内的局部值，不另做存活判定。
        """
        request_type = type(request)
        if request_type in _REQUEST_READS:
            if dispatch_topic is not None:
                raise TypeError(f"{request_type.__name__} does not take a dispatch topic")
            run = partial(self._handlers[request_type], request, actor=actor)
        elif request_type in _INTENT_REQUESTS:
            if dispatch_topic is None:
                raise TypeError(f"{request_type.__name__} requires a dispatch topic")
            run = partial(
                self._handlers[request_type], request, actor=actor, dispatch_topic=dispatch_topic
            )
        else:
            raise TypeError(f"Unsupported request-level operation: {request_type.__name__}")
        result = await self._events.observe(
            operation_of(request), request, run, actor=actor, entry="request"
        )
        return cast(R, result)

    async def derive_child_thread(
        self, request: DeriveCallThreadRequest, *, thread: ThreadHandle
    ) -> ExecutionThread:
        """CALL 派生：确认调用方存活，以它授权 ``agent.call`` 并解析图纸，再交给线程表派生。

        调用方只能唤起它看得见的 Agent，这是调用方自己的 actor 操作。图纸冷读
        之后由线程表重新判定调用方存活并确认它是主线程，再为被调用方做准入、
        读取操作快照并登记；任一步拒绝都不留下登记。
        """
        resolved = self._threads.resolve(thread)
        caller = resolved.actor

        async def derive() -> ExecutionThread:
            profile = await self._agent.resolve_call_target(request, actor=caller)
            return self._threads.derive_child(parent=thread, profile=profile)

        return await self._events.observe(
            WorkspaceOperation.AGENT_CALL,
            request,
            derive,
            actor=caller,
            entry="thread",
            process_id=resolved.process_id,
        )

    def end_thread(self, *, thread: ThreadHandle) -> None:
        """同步结束子线程；已结束的线程是空操作，主线程以接线缺陷拒绝。"""
        self._threads.end_child(thread)


__all__ = ["WorkspaceOperationEntry"]
