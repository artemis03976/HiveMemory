"""Patchouli 交互提交 application 用例（父计划 5.6.5 例三 / 5.7.1）。

公开的独立交互提交入口：封装 ``InteractionSubmissionQueue`` 的接纳与
收据，绑定 ``interaction.submit`` operation。队列继续是 Patchouli 的
内部协作者——Passive/Alice/外部 adapter 统一经本用例提交，不直接持有
queue；单条事件接收、队列接纳、交互应用与 Memory 物化是不同事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.patchouli.control.interaction_submission import (
    InteractionOrigin,
    InteractionSubmission,
    InteractionSubmissionQueue,
)
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)

if TYPE_CHECKING:
    from hivememory.core.models import IdentityScope
    from hivememory.core.protocol.models import InteractionPayload


@dataclass(frozen=True)
class InteractionSubmitResult:
    """交互提交收据投影：请求已被队列接纳，应用进度按 interaction_id 跟踪。"""

    interaction_id: str
    work_id: str
    state: str


class InteractionSubmissionService:
    """经 Patchouli 交互队列的公开提交用例（``interaction.submit``）。"""

    def __init__(self, *, interaction_queue: InteractionSubmissionQueue) -> None:
        self._queue = interaction_queue

    async def submit_interaction(
        self,
        *,
        access: WorkspaceAccessContext,
        payload: InteractionPayload,
        identity_scope: IdentityScope | None = None,
        requested_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        origin: InteractionOrigin = "workspace_port",
    ) -> InteractionSubmitResult:
        """提交一份已发生的交互载荷；队列按 interaction_id 幂等接纳。

        ``identity_scope`` 是迁移期兼容参数：提供时必须与 access 上下文
        一致（DTO 不得覆盖可信坐标）。授权延迟的 flush 场景应在领域提交
        时取得或确认有效 access，不能把早先事件的 grant 当作无限期权限。
        """
        context = require_access_context(access, operation=WorkspaceOperation.INTERACTION_SUBMIT)
        scope = context.identity_scope
        if identity_scope is not None and identity_scope != scope:
            raise WorkspaceMismatchError(
                details={
                    "reason": "request_scope_mismatches_access_context",
                    "access_workspace_id": scope.workspace_identity.workspace_id,
                }
            )
        if payload is None:
            raise ValueError("submit_interaction 需要 payload 载荷")

        # 默认 interaction_id 必须每次唯一（uuid 派生）；需要重试语义的
        # 调用方显式提供稳定 interaction_id，由队列按其幂等去重。
        resolved_interaction_id = interaction_id or f"interaction_{uuid4().hex[:12]}"
        submission = InteractionSubmission(
            identity_scope=scope,
            interaction_id=resolved_interaction_id,
            payload=payload,
            requested_topic_id=requested_topic_id,
            ordering_key=f"topic:{requested_topic_id}",
            origin=origin,
            correlation={
                "principal_id": context.principal.principal_id,
                "submitted_by": scope.actor_identity.agent_id,
            },
        )
        receipt = await self._queue.submit(submission)
        return InteractionSubmitResult(
            interaction_id=receipt.interaction_id,
            work_id=receipt.work_id,
            state=receipt.state.value,
        )


__all__ = [
    "InteractionSubmissionService",
    "InteractionSubmitResult",
]
