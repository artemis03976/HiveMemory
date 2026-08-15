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


class WorkPayloadCodecError(WorkQueueError):
    """work payload 编解码契约失败。"""


class DuplicateWorkPayloadCodecError(WorkPayloadCodecError):
    """同一 kind 与 schema version 被重复注册。"""


class UnknownWorkPayloadCodecError(WorkPayloadCodecError):
    """work item 指向未注册的 payload codec。"""

    def __init__(self, kind: str, schema_version: int) -> None:
        self.kind = kind
        self.schema_version = schema_version
        super().__init__(
            f"Work payload codec '{kind}' schema version {schema_version} is not registered"
        )


class WorkPayloadEncodeError(WorkPayloadCodecError):
    """业务 payload 无法编码为稳定 JSON bytes。"""

    def __init__(self, kind: str, schema_version: int) -> None:
        self.kind = kind
        self.schema_version = schema_version
        super().__init__(
            f"Work payload codec '{kind}' schema version {schema_version} failed to encode"
        )


class WorkPayloadDecodeError(WorkPayloadCodecError):
    """JSON bytes 无法恢复为业务 payload。"""

    def __init__(self, kind: str, schema_version: int) -> None:
        self.kind = kind
        self.schema_version = schema_version
        super().__init__(
            f"Work payload codec '{kind}' schema version {schema_version} failed to decode"
        )


__all__ = [
    "DuplicateWorkItemError",
    "DuplicateWorkLaneError",
    "DuplicateWorkPayloadCodecError",
    "UnknownWorkLaneError",
    "UnknownWorkPayloadCodecError",
    "WorkPayloadCodecError",
    "WorkPayloadDecodeError",
    "WorkPayloadEncodeError",
    "WorkQueueCapacityError",
    "WorkQueueError",
    "WorkQueueStoppedError",
    "WorkStateConflictError",
]
