"""Local Work Queue Runtime 的公共契约与单机运行时。

queue 决定 work item 何时获得执行资源；scheduler 只决定何时触发 enqueue。
本包不接管 scheduler；具体 store adapter 位于 infrastructure 层。
"""

from hivememory.system.runtime.work_queue.cancellation import WorkCancellationToken
from hivememory.system.runtime.work_queue.exceptions import (
    DuplicateWorkItemError,
    DuplicateWorkLaneError,
    UnknownWorkLaneError,
    UnsupportedWorkQueueFeatureError,
    WorkQueueCapacityError,
    WorkQueueError,
    WorkQueueStoppedError,
    WorkStateConflictError,
)
from hivememory.system.runtime.work_queue.models import (
    TERMINAL_WORK_STATES,
    WORK_STATE_TRANSITIONS,
    WorkErrorSnapshot,
    WorkExecutionContext,
    WorkItem,
    WorkLaneShutdownSummary,
    WorkLaneSnapshot,
    WorkQueueShutdownSummary,
    WorkReceipt,
    WorkRecord,
    WorkState,
    can_transition_work_state,
)
from hivememory.system.runtime.work_queue.policies import (
    FailureAction,
    FailureDecision,
    QueuePolicy,
)
from hivememory.system.runtime.work_queue.ports import (
    WorkHandler,
    WorkHandlerPort,
    WorkQueue,
    WorkQueuePort,
    WorkStore,
    WorkStorePort,
)
from hivememory.system.runtime.work_queue.runtime import WorkLane, WorkQueueRuntime
from hivememory.system.runtime.work_queue.supervisor import WorkQueueSupervisor

__all__ = [
    "TERMINAL_WORK_STATES",
    "WORK_STATE_TRANSITIONS",
    "FailureAction",
    "FailureDecision",
    "QueuePolicy",
    "DuplicateWorkItemError",
    "DuplicateWorkLaneError",
    "UnknownWorkLaneError",
    "UnsupportedWorkQueueFeatureError",
    "WorkCancellationToken",
    "WorkErrorSnapshot",
    "WorkExecutionContext",
    "WorkHandler",
    "WorkHandlerPort",
    "WorkItem",
    "WorkLane",
    "WorkLaneShutdownSummary",
    "WorkLaneSnapshot",
    "WorkQueue",
    "WorkQueueCapacityError",
    "WorkQueueError",
    "WorkQueuePort",
    "WorkQueueRuntime",
    "WorkQueueShutdownSummary",
    "WorkQueueStoppedError",
    "WorkReceipt",
    "WorkRecord",
    "WorkState",
    "WorkStore",
    "WorkStorePort",
    "WorkQueueSupervisor",
    "WorkStateConflictError",
    "can_transition_work_state",
]
