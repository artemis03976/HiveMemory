"""Patchouli 交互提交 application API（公开路由 ``patchouli.public.interaction.*``）。

公开的独立交互提交入口：封装 ``InteractionSubmissionQueue`` 的接纳、收据
与进度查询。队列继续是 Patchouli 的内部协作者——Import Bus 等外部来源统一
经本 API 提交，不直接持有 queue；单条事件接收、队列接纳、交互应用与
Memory 物化是不同事实。

交互提交属于系统编排，不属于 actor 操作：本层只接收授权点传入的
``IdentityScope``，在公共边界拆为归属与发起者。
"""

from __future__ import annotations

from uuid import uuid4

from hivememory.components.work_queue import (
    TERMINAL_WORK_STATES,
    WorkQueueCapacityError,
    WorkQueueStoppedError,
    WorkState,
)
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.patchouli.contracts.interaction_submission import (
    InteractionOrigin,
    InteractionProgress,
    InteractionStatus,
    InteractionSubmitResult,
)
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
)


class InteractionSubmissionService:
    """经 Patchouli 交互队列的公开提交与进度查询 API。"""

    def __init__(self, *, interaction_queue: InteractionSubmissionQueue) -> None:
        self._queue = interaction_queue

    async def submit_interaction(
        self,
        *,
        identity_scope: IdentityScope,
        payload: InteractionPayload,
        requested_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        origin: InteractionOrigin = "workspace_port",
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
    ) -> InteractionSubmitResult:
        """提交一份已发生的交互载荷；队列按 interaction_id 幂等接纳。

        ``identity_scope`` 由调用方的授权点组装后传入。新话题的标题与摘要
        只在目标为 ``NEW_TOPIC`` 时允许非空。队列容量已满或已停止表示
        这次提交没有被接管，以 ``InteractionAdmissionUnavailableError``
        拒绝，调用方可用同一 interaction_id 重试。
        """
        scope = require_identity_scope(identity_scope)
        if payload is None:
            raise ValueError("submit_interaction 需要 payload 载荷")

        # 默认 interaction_id 必须每次唯一（uuid 派生）；需要重试语义的
        # 调用方显式提供稳定 interaction_id，由队列按其幂等去重。
        resolved_interaction_id = interaction_id or f"interaction_{uuid4().hex[:12]}"
        submission = InteractionSubmission(
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id=resolved_interaction_id,
            payload=payload,
            requested_topic_id=requested_topic_id,
            ordering_key=f"topic:{requested_topic_id}",
            origin=origin,
            correlation={},
            new_topic_title=new_topic_title,
            new_topic_summary=new_topic_summary,
        )
        try:
            receipt = await self._queue.submit(submission)
        except WorkQueueCapacityError as error:
            raise InteractionAdmissionUnavailableError(reason="capacity_rejected") from error
        except WorkQueueStoppedError as error:
            raise InteractionAdmissionUnavailableError(reason="queue_stopped") from error
        return InteractionSubmitResult(
            interaction_id=receipt.interaction_id,
            work_id=receipt.work_id,
            state=receipt.state.value,
        )

    async def interaction_status(
        self, *, identity_scope: IdentityScope, interaction_id: str
    ) -> InteractionStatus | None:
        """回答该交互在提交队列中的进度，供提交方在重发前查重、续接上一轮的落点。

        归属是硬边界：同名交互已由另一个 Workspace 提交时以
        ``WorkspaceMismatchError`` 拒绝，不把其他 Workspace 的接纳当作重复。
        查询只覆盖队列的保留窗口，窗口外的终态记录视为未接纳。
        """
        scope = require_identity_scope(identity_scope)
        view = await self._queue.lookup(interaction_id)
        if view is None:
            return None
        if view.belong_to != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"interaction_id": interaction_id})
        return InteractionStatus(
            interaction_id=interaction_id,
            state=_progress(view.state),
            topic_id=view.topic_id,
        )


def _progress(state: WorkState | None) -> InteractionProgress:
    """把通用工作状态归为提交方关心的四种进度。"""
    if state is None:
        return "unknown"
    if state is WorkState.SUCCEEDED:
        return "succeeded"
    if state in TERMINAL_WORK_STATES:
        return "failed"
    return "pending"


__all__ = [
    "InteractionSubmissionService",
]
