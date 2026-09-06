from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import (
    IdentityScope,
    TopicSnapshot,
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

    身份入口约定（v0.6.2 收敛）：Topic 管理是用户导向的管理用例，server
    边界为其冻结 ``system`` actor 的 IdentityScope；本层不再解析裸
    ``user_id``，只通过全局总线调用 Patchouli 公共能力，业务结果保持
    强类型，不在这里包装 HTTP 字典。
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

    async def list_active_topics(self, *, identity_scope: IdentityScope) -> tuple[TopicSnapshot, ...]:
        """在显式 Workspace scope 中列出活跃 Topic 快照。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            identity_scope=identity_scope,
        )

    async def settle_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str | None = None,
    ) -> TopicSettleResult:
        """结算 Topic，并原样返回 Patchouli 的业务结果。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
        )

    async def evict_topic(
        self,
        *,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicEvictionResult:
        """删除 Topic，并原样返回 Patchouli 的驱逐结果。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            identity_scope=identity_scope,
            topic_id=topic_id,
        )
