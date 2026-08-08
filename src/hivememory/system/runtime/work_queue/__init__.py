"""Local Work Queue Runtime 的 Q0 公共契约。

queue 决定 work item 何时获得执行资源；scheduler 只决定何时触发 enqueue。
本包不接管 scheduler，也不包含 worker/store adapter 实现。
"""

from hivememory.system.runtime.work_queue.models import (
    TERMINAL_WORK_STATES,
    WORK_STATE_TRANSITIONS,
    WorkErrorSnapshot,
    WorkExecutionContext,
    WorkItem,
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

__all__ = [
    "TERMINAL_WORK_STATES",
    "WORK_STATE_TRANSITIONS",
    "FailureAction",
    "FailureDecision",
    "QueuePolicy",
    "WorkErrorSnapshot",
    "WorkExecutionContext",
    "WorkHandler",
    "WorkHandlerPort",
    "WorkItem",
    "WorkQueue",
    "WorkQueuePort",
    "WorkReceipt",
    "WorkRecord",
    "WorkState",
    "WorkStore",
    "WorkStorePort",
    "can_transition_work_state",
]
