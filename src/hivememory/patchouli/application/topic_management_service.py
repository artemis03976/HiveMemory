from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import (
    IdentityScope,
    TopicData,
    TopicSnapshot,
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

    管理列表与生命周期变更由 workspace root API 授权；进程读取话题
    时由系统以 root 授权。本层是授权点以下的资源 owner，只接收授权点组装的
    ``IdentityScope``，不接收工作 actor。Topic 可见性仍由领域实现强制。

    ``prepare_topic`` 为一轮确认或建立话题，由 root API 为任务进程的写入与
    Import Bus 的写入时确定话题调用，不属于 Actor 行为目录。
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
        snapshots = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            belong_to=scope.workspace_identity,
            include_empty=include_empty,
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
        topic_id: str,
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
        *,
        identity_scope: IdentityScope,
        target_topic_id: str,
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
    ) -> str:
        """确认或建立目标话题（必要时先按 LRU 驱逐），返回真实话题标识。

        ``target_topic_id`` 为已有话题时确认它存在并刷新驻留；为 ``NEW_TOPIC``
        时以给出的标题与摘要建立新话题。归属取 scope 的 Workspace。
        """
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            scope.workspace_identity,
        )


__all__ = ["TopicManagementService"]
