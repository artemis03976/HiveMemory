"""Workspace 会话、写入意图与任务进程交互提交的公共错误。"""

from typing import Literal

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


ActiveFinalizationStage = Literal[
    "interaction_admission",
    "interaction_apply",
]


class ActiveInteractionFinalizationError(WorkspaceDomainError):
    """任务进程的交互提交未能跨过接纳或应用的成功边界；会话仍保存了这一轮。

    ``details`` 中 ``stage`` 区分没有被接纳与接纳后应用没有成功，``reason`` 是
    原因码，``error_class`` 是应用失败时的异常类名。机器码沿用交互提交原有的
    错误码，对客户端保持不变。
    """

    code = "patchouli.interaction.finalization_failed"

    def __init__(
        self,
        *,
        interaction_id: str,
        stage: ActiveFinalizationStage,
        reason: str,
        error_class: str | None = None,
    ) -> None:
        super().__init__(
            details={
                "interaction_id": interaction_id,
                "stage": stage,
                "reason": reason,
                "error_class": error_class,
            }
        )


__all__ = [
    "SessionNotFoundError",
    "SessionBusyError",
    "SessionCapacityError",
    "SessionCheckpointError",
    "SessionClosedError",
    "PendingUpdateNotAllowedError",
    "IntentDispatchRejectedError",
    "IntentDispatchUnknownError",
    "ActiveFinalizationStage",
    "ActiveInteractionFinalizationError",
]
