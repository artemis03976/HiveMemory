from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import (
    Identity,
    TopicSnapshot,
    resolve_default_identity_scope,
)
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class TopicApplicationService:
    """Topic HTTP 用例对应的系统应用服务。

    本层只负责把用户身份转换为默认 Workspace 访问上下文，并通过全局总线
    调用 Patchouli 公共能力；业务结果保持强类型，不在这里包装 HTTP 字典。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        config: HiveMemoryConfig,
    ) -> None:
        self._global_bus = global_bus
        self._config = config

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    async def list_active_topics(self, *, user_id: str) -> tuple[TopicSnapshot, ...]:
        """列出用户默认 Workspace 中的活跃 Topic 快照。"""

        identity = Identity(user_id=user_id)
        identity_scope = resolve_default_identity_scope(identity)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            identity_scope=identity_scope,
        )

    async def settle_topic(
        self,
        *,
        user_id: str,
        topic_id: str | None = None,
    ) -> TopicSettleResult:
        """结算 Topic，并原样返回 Patchouli 的业务结果。"""

        identity_scope = resolve_default_identity_scope(
            Identity(user_id=user_id),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
        )

    async def evict_topic(
        self,
        *,
        user_id: str,
        topic_id: str,
    ) -> TopicEvictionResult:
        """删除 Topic，并原样返回 Patchouli 的驱逐结果。"""

        identity_scope = resolve_default_identity_scope(
            Identity(user_id=user_id),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
        )
