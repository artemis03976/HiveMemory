"""root Topic API：授权后调用话题列表、结算与驱逐的公共路由，并为导入的一轮路由与准备话题。

路由与准备话题服务于 Import Bus：外部 harness 的一轮在 HiveMemory 没有任务进程，
它的话题路由（Gateway 的 PASSIVE_MEMORY 模式）与写入时的话题准备由系统执行，
每次调用都在入口 root 授权。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.protocol.gateway import GatewayIngressMode
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import TopicSnapshot, WorkspaceIdentity
    from hivememory.core.protocol.gateway import (
        GatewayDecision,
        GatewayProcessResult,
        GatewaySessionTurn,
    )
    from hivememory.patchouli.contracts import PreparedAgentRun
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootTopicService:
    """root 话题入口，生命周期结论沿用 Patchouli 的强类型公共结果。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer

    async def list_active_topics(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> tuple[TopicSnapshot, ...]:
        """列出活跃 Topic 快照（root 管理读取）。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.TOPIC_LIST_ACTIVE,
            identity_scope=scope,
        )

    async def settle_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        topic_id: str,
        actor: WorkingActor,
    ) -> TopicSettleResult:
        """结算 Topic（root 授权），并原样返回 Patchouli 的业务结果。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.MANUAL_SETTLE_TOPIC,
            identity_scope=scope,
            topic_id=topic_id,
        )

    async def evict_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        topic_id: str,
        actor: WorkingActor,
    ) -> TopicEvictionResult:
        """删除 Topic（root 授权），并原样返回 Patchouli 的驱逐结果。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.EVICT_TOPIC,
            identity_scope=scope,
            topic_id=topic_id,
        )

    async def route_passive_turn(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        message: str,
        session_turns: tuple[GatewaySessionTurn, ...],
        request_timeout_ms: int,
        actor: WorkingActor,
    ) -> GatewayProcessResult:
        """以 root 授权、以 PASSIVE_MEMORY 模式请求 Gateway 为一轮导入的交互做话题路由。

        Gateway 的结果与异常原样交回：PASSIVE_MEMORY 只给出话题路由与新话题的
        标题、摘要，不检索；结果的类型与可恢复异常由调用方判断。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            GatewayPublicRoutes.PROCESS,
            message=message,
            identity_scope=scope,
            ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
            request_timeout_ms=request_timeout_ms,
            session_turns=session_turns,
        )

    async def prepare_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        gateway_decision: GatewayDecision,
        actor: WorkingActor,
    ) -> PreparedAgentRun:
        """以 root 授权按 Gateway 的路由确认或新建一轮的话题，不检索。

        经 Patchouli prepare 完成，不给检索 scope；新话题在此建立（必要时先按
        LRU 驱逐），prepare 失败时由 Patchouli 清理它新建的空话题。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.PREPARE_AGENT_RUN,
            identity_scope=scope,
            interaction_id=interaction_id,
            gateway_decision=gateway_decision,
        )


__all__ = ["RootTopicService"]
