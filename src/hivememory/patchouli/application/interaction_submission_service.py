"""Patchouli 交互提交 application API（公开路由 ``patchouli.public.interaction.*``）。

公开的独立交互提交入口：封装 ``InteractionSubmissionQueue`` 的接纳、等待应用、
结局归类与进度查询。队列继续是 Patchouli 的内部协作者——任务进程与 Import Bus
都经 root API 调用本 API，不直接持有 queue；单条事件接收、队列接纳、交互应用
与 Memory 物化是不同事实。

交互提交是系统级操作，不属于 actor 操作：本层只接收授权点以 root 授权
组装的 ``IdentityScope`` 作为归属，交互的发起者取自显式的参与者。
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
from hivememory.core.models import ActorIdentity, IdentityScope, require_identity_scope
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
    topic_ordering_key,
)

_NEW_TOPIC = "NEW_TOPIC"


class InteractionSubmissionService:
    """经 Patchouli 交互队列的公开提交、等待与进度查询 API。"""

    def __init__(self, *, interaction_queue: InteractionSubmissionQueue) -> None:
        self._queue = interaction_queue

    async def submit_interaction(
        self,
        *,
        identity_scope: IdentityScope,
        participant: ActorIdentity,
        payload: InteractionPayload,
        requested_topic_id: str = _NEW_TOPIC,
        interaction_id: str | None = None,
        origin: InteractionOrigin = "workspace_port",
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
        timeout_seconds: float | None = None,
    ) -> InteractionSubmitResult:
        """提交一份已发生的交互载荷并等待它应用，返回结局。

        ``identity_scope`` 由调用方的授权点组装后传入，只提供归属；
        ``participant`` 是交互中的 Agent，记为发起者。新话题的标题与摘要
        只在目标为 ``NEW_TOPIC`` 时允许非空。队列按 interaction_id 幂等接纳：
        同一交互仍在应用或已应用时不重复入队，只等待已有的工作项；应用失败
        后允许以同一标识换载荷重新提交。

        ``timeout_seconds`` 为空时等到终态，上限由队列自身的应用时限与重试
        决定；到时限仍未应用完成时结局为 ``pending``，调用方可用同一标识稍后
        重发。接纳失败、应用失败与超时只在这里归类：容量已满或队列已停止
        表示这次提交没有被接管，以 ``InteractionAdmissionUnavailableError``
        拒绝；应用失败与超时以结局交回。
        """
        scope = require_identity_scope(identity_scope)
        if payload is None:
            raise ValueError("submit_interaction 需要 payload 载荷")

        # 默认 interaction_id 必须每次唯一（uuid 派生）；需要重试语义的
        # 调用方显式提供稳定 interaction_id，由队列按其幂等去重。
        resolved_interaction_id = interaction_id or f"interaction_{uuid4().hex[:12]}"
        submission = InteractionSubmission(
            belong_to=scope.workspace_identity,
            from_actor=participant,
            interaction_id=resolved_interaction_id,
            payload=payload,
            requested_topic_id=requested_topic_id,
            ordering_key=topic_ordering_key(requested_topic_id),
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

        outcome = await self._queue.wait(receipt, timeout=timeout_seconds)
        if outcome is None:
            # 刚接纳的交互的旁路索引只在终态后才可能被裁剪，查不到属于队列缺陷。
            raise RuntimeError(f"已接纳的交互在提交队列中查不到: {resolved_interaction_id}")
        state = _progress(outcome.state)
        return InteractionSubmitResult(
            interaction_id=resolved_interaction_id,
            state=state,
            topic_id=outcome.topic_id,
            error_class=outcome.error_class if state == "failed" else None,
            new_topic=requested_topic_id == _NEW_TOPIC,
        )

    async def interaction_status(
        self, *, identity_scope: IdentityScope, interaction_id: str
    ) -> InteractionStatus | None:
        """回答该交互在提交队列中的进度，供提交前查重、续接上一轮所在的话题。

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

    async def wait_interaction(
        self,
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        timeout_seconds: float | None,
    ) -> InteractionStatus | None:
        """等待已接纳的交互应用到终态或时限，返回此时的进度；未被接纳时返回 ``None``。

        供提交前查重时遇到仍在应用的交互：成功时进度带实际所在的话题，超时
        时仍为 ``pending``，队列停止时返回停止那一刻的进度；时限为空时等到
        终态。归属校验与 ``interaction_status`` 相同，先于等待完成。
        """
        status = await self.interaction_status(
            identity_scope=identity_scope, interaction_id=interaction_id
        )
        if status is None or status.state != "pending":
            return status
        await self._queue.wait(interaction_id, timeout=timeout_seconds)
        return await self.interaction_status(
            identity_scope=identity_scope, interaction_id=interaction_id
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
