"""Workspace 会话与写入意图的公共错误。"""

from hivememory.core.errors import WorkspaceDomainError


class SessionNotFoundError(WorkspaceDomainError):
    """会话不存在或不属于目标 Workspace。"""

    code = "workspace.session.not_found"


class SessionBusyError(WorkspaceDomainError):
    """会话已被运行中的进程认领。"""

    code = "workspace.session.busy"


class SessionCapacityError(WorkspaceDomainError):
    """会话或轮数达到容量上限。"""

    code = "workspace.session.capacity_exceeded"


class SessionCheckpointError(WorkspaceDomainError):
    """会话压缩检查点的覆盖边界不合法。"""

    code = "workspace.session.checkpoint_invalid"


class SessionClosedError(WorkspaceDomainError):
    """会话服务已关闭，拒绝继续读写。"""

    code = "workspace.session.closed"


class PendingUpdateNotAllowedError(WorkspaceDomainError):
    """UPDATE 的基础仍是待物化意图，不能作为正式原子更新。"""

    code = "workspace.memory.intent.pending_update_not_allowed"


class IntentDispatchRejectedError(WorkspaceDomainError):
    """生成队列明确拒绝本次意图，登记句柄已标记 FAILED。"""

    code = "workspace.memory.intent.dispatch_rejected"


class IntentDispatchUnknownError(WorkspaceDomainError):
    """无法确认派发结果，句柄保留 MATERIALIZING 等待可能的结算。"""

    code = "workspace.memory.intent.dispatch_unknown"


__all__ = [
    "SessionNotFoundError",
    "SessionBusyError",
    "SessionCapacityError",
    "SessionCheckpointError",
    "SessionClosedError",
    "PendingUpdateNotAllowedError",
    "IntentDispatchRejectedError",
    "IntentDispatchUnknownError",
]
