"""root Topic 管理 API：授权后调用话题列表、结算与驱逐的公共路由。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import TopicSnapshot, WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootTopicService:
    """root 话题管理入口，生命周期结论沿用 Patchouli 的强类型公共结果。"""

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
        access: WorkspaceAccessContext,
    ) -> tuple[TopicSnapshot, ...]:
        """列出活跃 Topic 快照（root 管理读取）。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.TOPIC_LIST_ACTIVE,
            identity_scope=scope,
        )

    async def settle_topic(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        topic_id: str,
        access: WorkspaceAccessContext,
    ) -> TopicSettleResult:
        """结算 Topic（root 授权），并原样返回 Patchouli 的业务结果。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
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
        access: WorkspaceAccessContext,
    ) -> TopicEvictionResult:
        """删除 Topic（root 授权），并原样返回 Patchouli 的驱逐结果。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.EVICT_TOPIC,
            identity_scope=scope,
            topic_id=topic_id,
        )


__all__ = ["RootTopicService"]
