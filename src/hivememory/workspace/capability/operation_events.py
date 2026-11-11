"""actor 操作的结局事件：操作入口分派的每一次 actor 操作在结束时发一个 ``actor.operation``。

事件只携带操作种类、入口、耗时与按操作的摘要（条数、待生成记忆的别名、被调用的
Agent），不携带检索文本、记忆正文或请求参数。工作 actor 不进入事件，只写它的
Agent 与驻留 Workspace 两个标签。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any, Literal

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkingActor, WorkspaceOperation
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors.base import DomainError
from hivememory.workspace.contracts.operations import (
    OperationRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)

type OperationEntryKind = Literal["thread", "request"]

_OPERATIONS: dict[type[OperationRequest[Any]], WorkspaceOperation] = {
    RetrieveRequest: WorkspaceOperation.MEMORY_SEARCH,
    ResolveReferencesRequest: WorkspaceOperation.MEMORY_READ,
    ResolveRunTargetRequest: WorkspaceOperation.MEMORY_RUN,
    SubmitWriteIntentRequest: WorkspaceOperation.MEMORY_WRITE,
    SubmitUpdateIntentRequest: WorkspaceOperation.MEMORY_UPDATE,
}

# 解析为正式记忆或重定向到正式记忆的引用，才算交付给了调用方。
_DELIVERED_KINDS = frozenset({"atom", "redirect"})


def operation_of(request: OperationRequest[Any]) -> WorkspaceOperation:
    """请求对应的 actor 操作；CALL 派生不经请求对象，由入口直接给出。"""
    return _OPERATIONS[type(request)]


def _summary(operation: WorkspaceOperation, request: object, result: Any) -> dict[str, object]:
    """按操作取结果的摘要：只有数量、解析结果种类、别名与 Agent 标识。"""
    if operation is WorkspaceOperation.MEMORY_SEARCH:
        return {"result_count": len(result)}
    if operation is WorkspaceOperation.MEMORY_READ:
        assert isinstance(request, ResolveReferencesRequest)
        delivered = sum(1 for resolution in result if resolution.kind in _DELIVERED_KINDS)
        return {"requested_count": len(request.aliases), "delivered_count": delivered}
    if operation is WorkspaceOperation.MEMORY_RUN:
        return {"resolution": result.kind}
    if operation in (WorkspaceOperation.MEMORY_WRITE, WorkspaceOperation.MEMORY_UPDATE):
        return {"pending_alias": result.pending_alias}
    return {"callee_agent_id": result.descriptor.labels.agent_id}


class ActorOperationEventEmitter:
    """包住一次 actor 操作的分派，在它结束时发出结局事件。

    结局放在 ``status``：返回为 ``completed``；领域错误为 ``rejected``，``reason`` 是
    错误码，错误带受约束的 ``details["reason"]`` 时另放进 ``data.error_reason``（如
    ``operation_not_allowed`` 与 ``root_required`` 同为 ``workspace.operation_denied``），
    ``details`` 的其余字段不进入事件；其他异常为 ``failed``，``data.error_class`` 是
    异常类名。异常与取消都原样传播，取消不发事件。事件只是旁路观测，发布失败由
    publisher 吞下。
    """

    def __init__(self, publisher: RuntimeEventPublisher | None = None) -> None:
        self._events = (publisher or RuntimeEventPublisher(NullRuntimeEventSink())).scoped(
            subsystem="workspace", component="operation_entry"
        )

    async def observe[R](
        self,
        operation: WorkspaceOperation,
        request: object,
        run: Callable[[], Awaitable[R]],
        *,
        actor: WorkingActor,
        entry: OperationEntryKind,
        process_id: str | None = None,
    ) -> R:
        events = self._events.bind(
            agent_id=actor.actor.agent_id,
            workspace_id=actor.workspace.workspace_id,
            process_id=process_id,
        )
        data: dict[str, object] = {"operation": operation.value, "entry": entry}
        started = monotonic()
        try:
            result = await run()
        except DomainError as exc:
            data["duration_ms"] = _elapsed_ms(started)
            if (error_reason := exc.details.get("reason")) is not None:
                data["error_reason"] = error_reason
            events.emit(
                RuntimeEventType.ACTOR_OPERATION,
                status="rejected",
                severity="warning",
                reason=exc.code,
                data=data,
            )
            raise
        except Exception as exc:
            data["duration_ms"] = _elapsed_ms(started)
            data["error_class"] = type(exc).__name__
            events.emit(
                RuntimeEventType.ACTOR_OPERATION,
                status="failed",
                severity="error",
                reason="actor_operation_failed",
                data=data,
            )
            raise
        data["duration_ms"] = _elapsed_ms(started)
        data.update(_summary(operation, request, result))
        events.emit(RuntimeEventType.ACTOR_OPERATION, status="completed", data=data)
        return result


def _elapsed_ms(started: float) -> float:
    return round((monotonic() - started) * 1000, 3)


__all__ = ["ActorOperationEventEmitter", "OperationEntryKind", "operation_of"]
