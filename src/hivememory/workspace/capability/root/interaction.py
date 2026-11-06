"""root 交互提交 API：授权后把一份已发生的交互提交给 Patchouli，并查询它的应用进度。

交互提交是系统级操作，Agent 不能拥有这项能力：执行者是保留的 ``system``，
交互中的 Agent 作为参与者显式传递，记为交互的发起者。外部 harness 的导入经
Import Bus 编排后由它提交，提交前的查重、续接上一轮所在的话题与提交后的等待
应用也经这里，每次调用都在入口 root 授权。任务进程的 finalize 依赖本轮 prepare
的结果，走 Patchouli 的 finalize 路由，同样以 root 授权，不经这里。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import ActorIdentity, WorkspaceIdentity
    from hivememory.core.protocol.models import InteractionPayload
    from hivememory.patchouli.contracts.interaction_submission import (
        InteractionOrigin,
        InteractionStatus,
        InteractionSubmitResult,
    )
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootInteractionService:
    """只向保留的 system actor 提供的交互提交与进度入口；接纳与幂等由 Patchouli 的提交队列负责。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer

    async def submit_interaction(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        payload: InteractionPayload,
        requested_topic_id: str,
        interaction_id: str,
        origin: InteractionOrigin,
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
        actor: WorkingActor,
    ) -> InteractionSubmitResult:
        """以 root 授权提交一份已封口的交互（root 系统级操作）。

        ``participant`` 是交互中的 Agent，由调用方从它的认证结果取得；scope
        只提供归属。队列按 ``interaction_id`` 幂等接纳，容量已满或已停止以
        ``InteractionAdmissionUnavailableError`` 拒绝，可用同一标识重试。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
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
        )

    async def interaction_status(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        actor: WorkingActor,
    ) -> InteractionStatus | None:
        """以 root 授权查询一份交互在提交队列中的进度；未被接纳时返回 ``None``。

        成功时进度带实际所在的话题；同名交互属于其他 Workspace 时由 Patchouli
        以冲突拒绝。查询只覆盖队列的保留窗口。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.INTERACTION_STATUS,
            identity_scope=scope,
            interaction_id=interaction_id,
        )

    async def wait_interaction(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        timeout_seconds: float,
        actor: WorkingActor,
    ) -> InteractionStatus | None:
        """以 root 授权等待一份已提交的交互应用到终态或超时，返回此时的进度。

        超时时进度仍为 ``pending``；未被接纳时返回 ``None``。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.INTERACTION_WAIT,
            identity_scope=scope,
            interaction_id=interaction_id,
            timeout_seconds=timeout_seconds,
        )


__all__ = ["RootInteractionService"]
