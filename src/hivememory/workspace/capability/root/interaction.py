"""root 交互提交 API：授权后把一轮已发生的交互提交给 Patchouli 并等待它应用。

交互提交是系统级操作，Agent 不能拥有这项能力：执行者是保留的 ``system``，
交互中的 Agent 作为参与者显式传递，记为交互的发起者。任务进程的结算与
Import Bus 的导入都经这里提交，方法入口 root 授权一次，提交前的查重、话题
路由与提交后的等待应用都在方法内部完成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.interaction_submission import (
    InteractionStatus,
    InteractionSubmitResult,
)
from hivememory.workspace.capability.root.passive_routing import (
    PassiveTurnRouting,
    route_passive_turn,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import ActorIdentity, WorkspaceIdentity
    from hivememory.core.protocol.gateway import GatewayDecision
    from hivememory.core.protocol.models import InteractionPayload
    from hivememory.patchouli.contracts.interaction_submission import InteractionOrigin
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer

_NEW_TOPIC = "NEW_TOPIC"


class RootInteractionService:
    """只向保留的 system actor 提供的交互提交入口；接纳、等待与结局归类由 Patchouli 负责。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        gateway_request_timeout_ms: int,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._gateway_request_timeout_ms = gateway_request_timeout_ms

    async def submit_interaction(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        payload: InteractionPayload,
        interaction_id: str,
        origin: InteractionOrigin,
        topic: str | GatewayDecision | PassiveTurnRouting,
        timeout_seconds: float | None = None,
        actor: WorkingActor,
    ) -> InteractionSubmitResult:
        """以 root 授权提交一轮已封口的交互并等待它应用，返回结局。

        ``topic`` 是话题来源：已确定的真实话题标识；Gateway 的决定，目标为
        ``NEW_TOPIC`` 时携带新话题的标题与摘要，话题在交互应用时建立；或一轮
        待路由的外部交互，由本方法以 PASSIVE_MEMORY 路由。``participant`` 是
        交互中的 Agent，scope 只提供归属。

        提交前先按 ``interaction_id`` 查进度：已应用或超出保留窗口的直接返回
        “此前已接纳”的结局；仍在应用的等待到终态或时限，等待的结果作为结局
        交回。两者都不再路由与提交——重新路由可能选中另一个话题，让同一交互的
        再次提交被队列判为冲突。只有未接纳或应用失败的交互才（重新）路由并提交。

        ``timeout_seconds`` 为空时等到终态。提交队列容量已满或已停止以
        ``InteractionAdmissionUnavailableError`` 拒绝，Gateway 的异常原样传出，
        两者都可用同一标识重试。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        status = await self._global_bus.request(
            PatchouliRoutes.INTERACTION_STATUS,
            identity_scope=scope,
            interaction_id=interaction_id,
        )
        if status is not None and status.state == "pending":
            status = await self._global_bus.request(
                PatchouliRoutes.INTERACTION_WAIT,
                identity_scope=scope,
                interaction_id=interaction_id,
                timeout_seconds=timeout_seconds,
            )
            if status is None:
                raise RuntimeError(f"已接纳的交互在提交队列中查不到: {interaction_id}")
            return _previously_accepted(status)
        if status is not None and status.state != "failed":
            return _previously_accepted(status)

        if isinstance(topic, PassiveTurnRouting):
            topic = await route_passive_turn(
                self._global_bus,
                scope,
                topic,
                request_timeout_ms=self._gateway_request_timeout_ms,
            )
        if isinstance(topic, str):
            requested_topic_id, new_topic_title, new_topic_summary = topic, None, None
        else:
            requested_topic_id = topic.target_topic_id
            # 新话题以 Gateway 给出的标题与摘要建立，后续路由依赖它们。
            is_new_topic = requested_topic_id == _NEW_TOPIC
            new_topic_title = topic.new_topic_title if is_new_topic else None
            new_topic_summary = topic.new_topic_summary if is_new_topic else None
        return await self._global_bus.request(
            PatchouliRoutes.INTERACTION_SUBMIT,
            identity_scope=scope,
            participant=participant,
            payload=payload,
            requested_topic_id=requested_topic_id,
            interaction_id=interaction_id,
            origin=origin,
            new_topic_title=new_topic_title,
            new_topic_summary=new_topic_summary,
            timeout_seconds=timeout_seconds,
        )


def _previously_accepted(status: InteractionStatus) -> InteractionSubmitResult:
    """本次调用之前已被接纳的交互的结局，取自它此刻的进度。"""
    return InteractionSubmitResult(
        interaction_id=status.interaction_id,
        state=status.state,
        topic_id=status.topic_id,
        previously_accepted=True,
    )


__all__ = ["RootInteractionService"]
