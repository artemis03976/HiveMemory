"""本地工作队列的结构化任务适配器与类型化句柄。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from hivememory.system.runtime.work_queue.models import (
    TERMINAL_WORK_STATES,
    WorkItem,
    WorkRecord,
    WorkState,
)
from hivememory.system.runtime.work_queue.payloads import WorkPayloadCodecRegistry
from hivememory.system.runtime.work_queue.ports import WorkQueuePort

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class QueueTaskIdentity:
    """由结构化任务定义派生出的队列元数据。"""

    work_id: str
    ordering_key: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    priority: int = 0


@runtime_checkable
class QueueTask(Protocol):
    """可入队任务定义需要实现的最小标识契约。"""

    task_id: str


@runtime_checkable
class QueueTaskAdapter(Protocol[TaskT]):
    """在结构化任务类型与队列私有信封之间进行转换。

    业务任务不需要继承运行时模型。每种任务只需提供稳定标识以及可版本化的
    编解码规则，进入运行时时再由适配器生成私有 ``WorkItem``。
    """

    kind: str
    schema_version: int

    def identity(self, task: TaskT) -> QueueTaskIdentity: ...

    def encode(self, task: TaskT) -> object: ...

    def decode(self, payload: object) -> TaskT: ...


def adapt_queue_task(
    task: TaskT,
    *,
    lane: str,
    adapter: QueueTaskAdapter[TaskT],
    codecs: WorkPayloadCodecRegistry,
) -> WorkItem:
    """为一个结构化任务创建仅供运行时使用的 ``WorkItem``。

    任务标识和调度元数据来自适配器，业务内容则先经过 codec registry 编码，
    因而通用运行时不需要理解具体任务类型。
    """

    identity = adapter.identity(task)
    payload = codecs.encode(adapter.kind, adapter.schema_version, task)
    return WorkItem(
        work_id=identity.work_id,
        lane=lane,
        kind=adapter.kind,
        schema_version=adapter.schema_version,
        payload=payload,
        ordering_key=identity.ordering_key,
        priority=identity.priority,
        correlation_id=identity.correlation_id,
        idempotency_key=identity.idempotency_key,
    )


@dataclass(frozen=True)
class TaskOutcome[ResultT]:
    """组合 ``WorkRecord`` 与进程内补充信息的类型化视图。

    ``record`` 是唯一执行状态真相；``result``、原始错误文本和取消原因只补充
    通用持久化结构未承载的领域信息，不能反向决定工作状态。
    """

    record: WorkRecord
    result: ResultT | None = None
    error: str | None = None
    cancel_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.record.state in TERMINAL_WORK_STATES

    @property
    def succeeded(self) -> bool:
        return self.record.state == WorkState.SUCCEEDED


class TaskHandle[TaskT, ResultT]:
    """在不持有任务状态的前提下查询、等待和取消结构化任务。

    句柄通过 ``WorkQueuePort`` 读取 ``WorkRecord``，自身只保存无法进入通用
    record 的进程内类型化结果和控制信号。它适用于当前 in-memory runtime，
    不能被当作可持久化的任务状态源。
    """

    def __init__(
        self,
        *,
        task: TaskT,
        task_id: str,
        work_id: str,
        queue: WorkQueuePort,
    ) -> None:
        self._task = task
        self._task_id = task_id
        self._work_id = work_id
        self._queue = queue
        self._execution_result: ResultT | None = None
        self._last_execution_error: str | None = None
        self._execution_started = asyncio.Event()
        self._pending_cancel_reason: str | None = None
        self._accepted_cancel_reason: str | None = None

    @property
    def task(self) -> TaskT:
        return self._task

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def work_id(self) -> str:
        return self._work_id

    @property
    def cancel_requested(self) -> bool:
        return (
            self._pending_cancel_reason is not None
            or self._accepted_cancel_reason is not None
        )

    @property
    def cancel_reason(self) -> str | None:
        return self._accepted_cancel_reason or self._pending_cancel_reason

    async def snapshot(self) -> TaskOutcome[ResultT] | None:
        """读取当前状态快照；工作不存在时返回 ``None``。"""

        return self._to_outcome(await self._queue.get(self._work_id))

    async def wait(self, timeout: float | None = None) -> TaskOutcome[ResultT] | None:
        """等待工作进入终态，并返回当时的类型化结果视图。"""

        return self._to_outcome(
            await self._queue.wait(self._work_id, timeout=timeout)
        )

    async def wait_started(self) -> None:
        """无需轮询 ``WorkRecord``，等待处理器开始首次执行。"""

        await self._execution_started.wait()

    async def cancel(self, *, reason: str = "user_requested") -> bool:
        """向运行时请求取消，并仅在请求被接纳后保留取消原因。"""

        previous_reason = self._pending_cancel_reason
        # 运行时可能在 cancel 返回前完成终态投影，因此先暂存原因；若调用失败或
        # 被拒绝，再回滚到调用前状态，避免句柄暴露并未生效的取消请求。
        self._pending_cancel_reason = reason
        try:
            accepted = await self._queue.cancel(self._work_id, reason=reason)
        except BaseException:
            self._pending_cancel_reason = previous_reason
            raise
        if accepted:
            self._accepted_cancel_reason = reason
        else:
            self._pending_cancel_reason = previous_reason
        return accepted

    def _record_execution_result(self, result: ResultT) -> None:
        """记录 ``WorkRecord`` 通用结构未承载的类型化结果值。"""

        self._execution_result = result
        self._last_execution_error = None

    def _record_execution_started(self) -> None:
        """通知领域控制器 handler 已开始执行，避免轮询状态记录。"""

        self._execution_started.set()

    def _record_execution_error(self, error: Exception) -> None:
        """保留进程内原始错误文本，供领域投影使用。"""

        self._last_execution_error = str(error) or type(error).__name__

    @property
    def _cached_execution_result(self) -> ResultT | None:
        return self._execution_result

    def _to_outcome(self, record: WorkRecord | None) -> TaskOutcome[ResultT] | None:
        """把状态真相与进程内补充信息合成为只读结果视图。"""

        if record is None:
            return None
        result = self._execution_result if record.state == WorkState.SUCCEEDED else None
        error = None
        if record.state in {WorkState.FAILED, WorkState.DEAD_LETTER}:
            error = self._last_execution_error
            if error is None and record.last_error is not None:
                error = record.last_error.message or record.last_error.error_class
        return TaskOutcome(
            record=record,
            result=result,
            error=error,
            cancel_reason=self.cancel_reason,
        )


__all__ = [
    "QueueTask",
    "QueueTaskAdapter",
    "QueueTaskIdentity",
    "TaskHandle",
    "TaskOutcome",
    "adapt_queue_task",
]
