from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import IdentityScope, TopicData, TopicSnapshot
from hivememory.patchouli.application.access_consumption import verified_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class TopicManagementService:
    """Patchouli 对外提供的 Topic 管理应用服务。

    A1 计划（第 1.1/4.1 节）补齐的访问差额：Topic 公共入口此前仅按裸
    scope 处理、未接入统一行为检查，现按既有操作的显式绑定接入——

    - ``list_active_topics`` / ``get_topic_data``：``resource.read``（调用
      方可见的 Topic 工作集读取，可见性仍由领域实现强制）；
    - ``settle_topic`` / ``evict_topic``：``management.topic``（生命周期
      变更；不借用 ``management.memory`` 泛化放行）。

    未提供 ``access`` 的旧调用方（Topic 管理 HTTP 链路）在 A1 第 6 节
    兼容清单内按受信适配保持既有行为，A6 完成消费者切换后收紧。
    ``prepare_topic`` 是内部对话编排用例（未挂载公开路由），不属于
    Actor 行为目录。
    """

    def __init__(self, *, bus: PatchouliBus, access_guard: WorkspaceAccessGuard) -> None:
        # Topic public API 只通过 local bus 组合 topic primitives，不直接持有 familiar。
        self._bus = bus
        self._access_guard = access_guard

    async def list_active_topics(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        include_empty: bool = False,
    ) -> tuple[TopicSnapshot, ...]:
        scope = verified_scope(
            access,
            WorkspaceOperation.RESOURCE_READ,
            identity_scope,
            access_guard=self._access_guard,
        )
        kwargs = {"identity_scope": scope}
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
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        topic_id: str,
    ) -> TopicData | None:
        """无副作用读取调用方可见的完整话题数据。"""
        scope = verified_scope(
            access,
            WorkspaceOperation.RESOURCE_READ,
            identity_scope,
            access_guard=self._access_guard,
        )
        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
            identity_scope=scope,
        )
        if topic_data is not None and topic_data.workspace_identity != scope.workspace_identity:
            # 控制面同样隐藏越域资源，不能把下游异常结果升级为可见性泄漏。
            return None
        return topic_data

    async def settle_topic(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        topic_id: str | None = None,
    ) -> TopicSettleResult:
        """通过本地总线结算 Topic（management.topic），返回稳定业务结果。"""
        scope = verified_scope(
            access,
            WorkspaceOperation.MANAGEMENT_TOPIC,
            identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            scope,
            topic_id,
        )

    async def evict_topic(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        topic_id: str,
    ) -> TopicEvictionResult:
        """通过本地总线驱逐 Topic（management.topic），不触发记忆结算。"""
        scope = verified_scope(
            access,
            WorkspaceOperation.MANAGEMENT_TOPIC,
            identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_EVICT,
            scope,
            topic_id,
        )

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity_scope: IdentityScope,
    ) -> str:
        """内部对话编排用例（旧 Active 链路，未挂载公开路由）。

        不属于 A1 的 Actor 行为目录；旧职责在 A6 随 Alice 收缩逐项关闭
        （协调计划兼容规则 4）。
        """
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity_scope,
        )


__all__ = ["TopicManagementService"]
