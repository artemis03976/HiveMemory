"""root Topic API：授权后调用话题列表、结算与驱逐的公共路由，并为一轮确定话题。

确定话题服务于写入：写入派发需要这一轮最终所在的话题。任务进程按 Gateway 的
决定确认已有话题或在第一次写入时建立新话题；外部 harness 的一轮在 HiveMemory
没有任务进程，由系统在方法内以 PASSIVE_MEMORY 路由后建立。每次调用都在入口
root 授权一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)
from hivememory.workspace.capability.root.passive_routing import (
    PassiveTurnRouting,
    route_passive_turn,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import TopicSnapshot, WorkspaceIdentity
    from hivememory.core.protocol.gateway import GatewayDecision
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer

_NEW_TOPIC = "NEW_TOPIC"


@dataclass(frozen=True)
class DeterminedTopic:
    """为一轮确定的话题：真实话题标识，以及它是否为这一轮新建。"""

    topic_id: str
    is_new: bool


class RootTopicService:
    """root 话题入口，生命周期结论沿用 Patchouli 的强类型公共结果。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        gateway_request_timeout_ms: int = 8000,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._gateway_request_timeout_ms = gateway_request_timeout_ms

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

    @overload
    async def determine_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        topic_source: GatewayDecision,
        actor: WorkingActor,
    ) -> DeterminedTopic: ...

    @overload
    async def determine_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        topic_source: PassiveTurnRouting,
        actor: WorkingActor,
    ) -> DeterminedTopic | None: ...

    async def determine_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        interaction_id: str,
        topic_source: GatewayDecision | PassiveTurnRouting,
        actor: WorkingActor,
    ) -> DeterminedTopic | None:
        """以 root 授权为一轮确认或建立话题，返回真实话题；不检索。

        ``topic_source`` 是任务进程已有的 Gateway 决定，或一轮待路由的外部
        交互。后者先查这一轮的进度：已应用的返回它实际所在的话题；已提交、
        仍在应用的不返回话题（返回 ``None``），所在的话题要等应用完成才确定，
        此时重新路由会让写入与这一轮落在不同的话题；否则以 PASSIVE_MEMORY 路由。

        目标是已有话题时确认它存在并刷新驻留，是 ``NEW_TOPIC`` 时以决定中的
        标题与摘要建立（必要时先按 LRU 驱逐）。Gateway 与话题准备的异常原样
        传出。
        """
        scope = self._authorizer.authorize_root(actor, target_workspace)
        if isinstance(topic_source, PassiveTurnRouting):
            status = await self._global_bus.request(
                PatchouliRoutes.INTERACTION_STATUS,
                identity_scope=scope,
                interaction_id=interaction_id,
            )
            # 只有已应用的轮次带实际所在的话题。
            if status is not None and status.topic_id is not None:
                return DeterminedTopic(topic_id=status.topic_id, is_new=False)
            if status is not None and status.state == "pending":
                return None
            decision = await route_passive_turn(
                self._global_bus,
                scope,
                topic_source,
                request_timeout_ms=self._gateway_request_timeout_ms,
            )
        else:
            decision = topic_source
        is_new = decision.target_topic_id == _NEW_TOPIC
        topic_id = await self._global_bus.request(
            PatchouliRoutes.PREPARE_TOPIC,
            identity_scope=scope,
            target_topic_id=decision.target_topic_id,
            new_topic_title=decision.new_topic_title if is_new else None,
            new_topic_summary=decision.new_topic_summary if is_new else None,
        )
        return DeterminedTopic(topic_id=topic_id, is_new=is_new)


__all__ = ["DeterminedTopic", "RootTopicService"]
