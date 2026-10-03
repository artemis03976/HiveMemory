"""Topic 能力：Topic 管理用例薄委托（A2 §1.2，自 ``system/application`` 迁入）。

当前只转发 Patchouli Topic 公共路由，Topic 资料切片由 A3 在同一能力骨架上
扩展；operation 授权（``resource.read`` / ``management.topic``）在本层、
路由调用前执行（A1 访问边界返工第 4.3 节）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import (
    IdentityScope,
    TopicSnapshot,
)
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class TopicApplicationService:
    """Topic HTTP 用例对应的系统应用服务。

    身份入口约定（v0.6.2 收敛）：Topic 管理是用户导向的管理用例，server
    边界为其冻结 ``system`` actor 的 IdentityScope；本层不再解析裸
    ``user_id``，只通过全局总线调用 Patchouli 公共能力，业务结果保持
    强类型，不在这里包装 HTTP 字典。

    访问上下文约定（A1 访问边界返工第 4.3 节）：``access`` 为统一认证网关
    签发的可信 context；读取（list 绑定 ``resource.read``）与生命周期变更
    （settle/evict 绑定 ``management.topic``）的 operation 授权在本层、路由
    调用前执行，随后把 access 原样透传给 Patchouli Topic 公共路由。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        self._global_bus = global_bus
        self._access_guard = access_guard

    async def list_active_topics(
        self,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext,
    ) -> tuple[TopicSnapshot, ...]:
        """列出活跃 Topic 快照（``resource.read``）。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.RESOURCE_READ)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            identity_scope=identity_scope,
            access=access,
        )

    async def settle_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str | None = None,
        access: WorkspaceAccessContext,
    ) -> TopicSettleResult:
        """结算 Topic（``management.topic``），并原样返回 Patchouli 的业务结果。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.MANAGEMENT_TOPIC)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
            access=access,
        )

    async def evict_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str,
        access: WorkspaceAccessContext,
    ) -> TopicEvictionResult:
        """删除 Topic（``management.topic``），并原样返回 Patchouli 的驱逐结果。"""
        self._access_guard.authorize_operation(access, WorkspaceOperation.MANAGEMENT_TOPIC)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
            access=access,
        )
