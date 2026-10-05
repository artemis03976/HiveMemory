from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import (
    IdentityScope,
    TopicData,
    TopicSnapshot,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class TopicManagementService:
    """Patchouli 对外提供的 Topic 管理应用服务。

    ``list_active_topics`` / ``get_topic_data``（``resource.read``）与
    ``settle_topic`` / ``evict_topic``（``management.topic``）的行为授权
    已上移到 workspace 能力层或任务进程的阶段检查（A1 访问边界返工第
    4.4/4.6 节）；本层是授权点以下的资源 owner，只接收授权点组装的
    ``IdentityScope``，不接收访问 context。Topic 可见性仍由领域实现强制。

    ``prepare_topic`` 是内部对话编排用例（未挂载公开路由），不属于
    Actor 行为目录。
    """

    def __init__(self, *, bus: PatchouliBus) -> None:
        # Topic public API 只通过 local bus 组合 topic primitives，不直接持有 familiar。
        self._bus = bus

    async def list_active_topics(
        self,
        *,
        identity_scope: IdentityScope,
        include_empty: bool = False,
    ) -> tuple[TopicSnapshot, ...]:
        scope = require_identity_scope(identity_scope)
        kwargs: dict[str, object] = {"belong_to": scope.workspace_identity}
        if include_empty:
            kwargs["include_empty"] = True
        snapshots = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            **kwargs,
        )
        return tuple(snapshots)

    async def get_topic_data(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicData | None:
        """无副作用读取调用方可见的完整话题数据。"""
        scope = require_identity_scope(identity_scope)
        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
            belong_to=scope.workspace_identity,
        )
        if topic_data is not None and topic_data.workspace_identity != scope.workspace_identity:
            # 控制面同样隐藏越域资源，不能把下游异常结果升级为可见性泄漏。
            return None
        return topic_data

    async def settle_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str | None = None,
    ) -> TopicSettleResult:
        """通过本地总线结算 Topic（生命周期变更授权在能力层），返回稳定业务结果。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            scope.workspace_identity,
            topic_id,
        )

    async def evict_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicEvictionResult:
        """通过本地总线驱逐 Topic（生命周期变更授权在能力层），不触发记忆结算。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_EVICT,
            scope.workspace_identity,
            topic_id,
        )

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        belong_to: WorkspaceIdentity,
    ) -> str:
        """内部对话编排只按资源归属准备话题，不接收操作 scope。"""
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            belong_to,
        )


__all__ = ["TopicManagementService"]
