"""Local Work Queue Runtime 的显式失败契约。"""

from __future__ import annotations


class WorkQueueError(RuntimeError):
    """工作队列通用异常基类。"""


class UnknownWorkLaneError(WorkQueueError):
    """work item 指向未注册 lane。"""


class DuplicateWorkLaneError(WorkQueueError):
    """同名 lane 被重复注册。"""


class DuplicateWorkItemError(WorkQueueError):
    """同一 store 中已存在相同 work ID。"""


class WorkQueueCapacityError(WorkQueueError):
    """lane 已达到容量上限并明确拒绝新 work item。"""

    def __init__(self, lane: str, capacity: int) -> None:
        self.lane = lane
        self.capacity = capacity
        super().__init__(f"Work queue lane '{lane}' reached capacity {capacity}")


class WorkQueueStoppedError(WorkQueueError):
    """runtime 停止接收新 work item。"""


class WorkStateConflictError(WorkQueueError):
    """store 收到不符合状态机的迁移请求。"""


class UnsupportedWorkQueueFeatureError(WorkQueueError):
    """配置启用了当前阶段尚未实现的队列能力。"""


__all__ = [
    "DuplicateWorkItemError",
    "DuplicateWorkLaneError",
    "UnknownWorkLaneError",
    "UnsupportedWorkQueueFeatureError",
    "WorkQueueCapacityError",
    "WorkQueueError",
    "WorkQueueStoppedError",
    "WorkStateConflictError",
]
