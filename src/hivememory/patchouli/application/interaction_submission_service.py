"""Patchouli 交互提交 application API（interaction.submit）。

公开的独立交互提交入口：封装 ``InteractionSubmissionQueue`` 的接纳与
收据。队列继续是 Patchouli 的内部协作者——Passive/Alice/外部 adapter
统一经本 API 提交，不直接持有 queue；单条事件接收、队列接纳、交互应用
与 Memory 物化是不同事实。

本路由目前没有生产调用方，operation 授权（``interaction.submit``）在
workspace 能力层出现对应方法时进行（总 Idea 15.6，A1 访问边界返工第
4.6 节）；本层只接收调用方传入的 ``IdentityScope``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from hivememory.core.models import require_identity_scope
from hivememory.patchouli.control.interaction_submission import (
    InteractionOrigin,
    InteractionSubmission,
    InteractionSubmissionQueue,
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
    """经 Patchouli 交互队列的公开提交 API（``interaction.submit``）。"""

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
    ) -> InteractionSubmitResult:
        """提交一份已发生的交互载荷；队列按 interaction_id 幂等接纳。

        ``identity_scope`` 由调用方的授权点组装后传入；出现生产调用方前
        应先在能力层增加对应方法并执行 operation 授权。
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
