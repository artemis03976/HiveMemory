"""Gateway 两阶段上下文准备契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.models import IdentityScope, TopicData, TopicSnapshot
from hivememory.gateway.errors import RecoverableGatewayError
from hivememory.gateway.topic_context import render_topic_snapshots
from hivememory.patchouli.contracts import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext


class CandidateTopics(BaseModel):
    """话题路由前可见的候选话题及其提示词菜单。"""

    topic_snapshots: tuple[TopicSnapshot, ...] = Field(default_factory=tuple)
    active_topics_menu: str = ""

    model_config = ConfigDict(frozen=True)


class GatewayContextProvider(Protocol):
    """只负责读取 Gateway 所需上下文，不参与业务决策（也不做授权判断）。

    ``access`` 是调用方绑定的访问 context，原样传给 Patchouli 读取路由；
    无 Actor context 的调用方传 ``None``，读取失败由步骤 fallback 保守降级。
    """

    async def prepare_candidate_topics(
        self,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None,
    ) -> CandidateTopics: ...

    async def prepare_routed_topic(
        self,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None,
        topic_id: str,
    ) -> TopicData | None: ...


class GlobalBusGatewayContextProvider:
    """通过 Patchouli public route 实现两阶段上下文准备。"""

    def __init__(
        self,
        *,
        global_bus: GlobalSystemBus,
        include_empty_topics: bool = False,
    ) -> None:
        self._global_bus = global_bus
        self._include_empty_topics = include_empty_topics

    async def prepare_candidate_topics(
        self,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None,
    ) -> CandidateTopics:
        try:
            snapshots = await self._global_bus.request(
                PatchouliRoutes.TOPIC_LIST_ACTIVE,
                identity_scope=identity_scope,
                access=access,
                include_empty=self._include_empty_topics,
            )
        except Exception as exc:
            raise RecoverableGatewayError(f"Patchouli candidate topics 不可用: {exc}") from exc

        if not isinstance(snapshots, tuple) or not all(
            isinstance(snapshot, TopicSnapshot) for snapshot in snapshots
        ):
            raise TypeError("Patchouli TOPIC_LIST_ACTIVE 违反只读 tuple 契约")
        return CandidateTopics(
            topic_snapshots=snapshots,
            active_topics_menu=render_topic_snapshots(snapshots),
        )

    async def prepare_routed_topic(
        self,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None,
        topic_id: str,
    ) -> TopicData | None:
        if topic_id == "NEW_TOPIC":
            return None
        try:
            topic_data = await self._global_bus.request(
                PatchouliRoutes.TOPIC_GET_DATA,
                identity_scope=identity_scope,
                access=access,
                topic_id=topic_id,
            )
        except Exception as exc:
            raise RecoverableGatewayError(f"Patchouli routed topic 不可用: {exc}") from exc

        if topic_data is not None and not isinstance(topic_data, TopicData):
            raise TypeError("Patchouli TOPIC_GET_DATA 违反 TopicData 契约")
        return topic_data


__all__ = [
    "CandidateTopics",
    "GatewayContextProvider",
    "GlobalBusGatewayContextProvider",
]
