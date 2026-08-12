"""本地工作队列的依赖反转端口。

queue port 面向应用组件，store port 面向基础设施 adapter，handler port 由业务
组件实现。三者只共享通用 work 契约，不共享具体业务成功、失败或重试语义。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, TypeVar, runtime_checkable

from hivememory.system.runtime.work_queue.models import (
    WorkErrorSnapshot,
    WorkExecutionContext,
    WorkItem,
    WorkLaneSnapshot,
    WorkReceipt,
    WorkRecord,
)
from hivememory.system.runtime.work_queue.policies import FailureDecision, QueuePolicy

HandlerPayloadT = TypeVar("HandlerPayloadT", contravariant=True)
HandlerResultT = TypeVar("HandlerResultT", covariant=True)


@runtime_checkable
class WorkHandlerPort(Protocol[HandlerPayloadT, HandlerResultT]):
    """由业务 lane 提供的执行与失败分类端口。"""

    async def execute(
        self,
        payload: HandlerPayloadT,
        context: WorkExecutionContext,
    ) -> HandlerResultT: ...

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision: ...


@runtime_checkable
class WorkStorePort(Protocol):
    """保存 work 状态真相并提供原子迁移的基础设施端口。"""

    @property
    def is_durable(self) -> bool: ...

    def configure_lane(self, lane: str, policy: QueuePolicy) -> None: ...

    async def enqueue(self, item: WorkItem) -> WorkRecord: ...

    async def claim_ready(
        self,
        lane: str,
        *,
        limit: int,
        lease_seconds: float,
    ) -> list[WorkRecord]: ...

    async def mark_succeeded(
        self,
        work_id: str,
        result_ref: str | None = None,
    ) -> None: ...

    async def schedule_retry(
        self,
        work_id: str,
        *,
        available_at: datetime,
        error: WorkErrorSnapshot,
    ) -> None: ...

    async def mark_failed(
        self,
        work_id: str,
        error: WorkErrorSnapshot,
    ) -> None: ...

    async def mark_dead_lettered(
        self,
        work_id: str,
        error: WorkErrorSnapshot,
    ) -> None: ...

    async def cancel(self, work_id: str) -> bool: ...

    async def get(self, work_id: str) -> WorkRecord | None: ...

    async def wait(
        self,
        work_id: str,
        timeout: float | None = None,
    ) -> WorkRecord | None: ...

    async def wait_for_ready(self, lane: str, timeout: float) -> None: ...

    async def snapshot(self, lane: str) -> WorkLaneSnapshot: ...


@runtime_checkable
class WorkQueuePort(Protocol):
    """应用组件提交、查询和取消 work item 的运行时端口。"""

    async def enqueue(self, item: WorkItem) -> WorkReceipt: ...

    async def cancel(
        self,
        work_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool: ...

    async def get(self, work_id: str) -> WorkRecord | None: ...

    async def wait(
        self,
        work_id: str,
        timeout: float | None = None,
    ) -> WorkRecord | None: ...


# 无 Port 后缀的名称用于对应设计文档中的角色术语；Port 名称是代码中的规范名称。
WorkHandler = WorkHandlerPort
WorkQueue = WorkQueuePort
WorkStore = WorkStorePort


__all__ = [
    "WorkHandler",
    "WorkHandlerPort",
    "WorkQueue",
    "WorkQueuePort",
    "WorkStore",
    "WorkStorePort",
]
