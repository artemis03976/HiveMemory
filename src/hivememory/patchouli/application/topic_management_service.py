from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import TopicData, TopicSnapshot, IdentityScope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class TopicManagementService:
    """Patchouli 对外提供的 Topic 管理应用服务。"""

    def __init__(self, *, bus: PatchouliBus) -> None:
        # Topic public API 只通过 local bus 组合 topic primitives，不直接持有 familiar。
        self._bus = bus

    async def list_active_topics(
        self,
        *,
        identity_scope: IdentityScope,
        include_empty: bool = False,
    ) -> tuple[TopicSnapshot, ...]:
        kwargs = {"identity_scope": identity_scope}
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

        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
            identity_scope=identity_scope,
        )
        if (
            topic_data is not None
            and topic_data.workspace_identity != identity_scope.workspace_identity
        ):
            # 控制面同样隐藏越域资源，不能把下游异常结果升级为可见性泄漏。
            return None
        return topic_data

    async def settle_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str | None = None,
    ) -> TopicSettleResult:
        """通过本地总线结算 Topic，并返回稳定业务结果。"""

        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            identity_scope,
            topic_id,
        )

    async def evict_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicEvictionResult:
        """通过本地总线驱逐 Topic，不触发记忆结算。"""

        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_EVICT,
            identity_scope,
            topic_id,
        )

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity_scope: IdentityScope,
    ) -> str:
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity_scope,
        )
