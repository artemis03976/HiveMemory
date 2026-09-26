"""能力层作为 client 访问 Patchouli backing 读取路由（宪章 §4.4 / §5.3）。

实现 ``workspace.resolution.CanonicalReadBackend``：resolver 的 L2 冷读经
``GlobalSystemBus`` 调用 Patchouli 公共 backing 路由（第二层 client-server），
不形成递归，也不持有 Patchouli 的 Runtime、Service 或存储对象。

库不可达（backing 路由未挂载，如 Patchouli 未启动或已卸载）时抛出
``ResourceUnavailableError``，由调用方显式失败，不以空结果伪装成功；
存储错误与领域错误按原结构化错误传播。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import IdentityScope, MemoryAtom, ResolvedAgentProfile
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.workspace.access import WorkspaceAccessContext


class BusCanonicalReadBackend:
    """基于 ``GlobalSystemBus`` 的 Patchouli canonical/Profile 读取 backing。"""

    def __init__(self, global_bus: GlobalSystemBus) -> None:
        self._bus = global_bus

    async def read(
        self,
        memory_id: UUID,
        *,
        scope: IdentityScope,
        access: WorkspaceAccessContext | None,
    ) -> MemoryAtom | None:
        return await self._request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            memory_id,
            access=access,
            identity_scope=scope,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
        access: WorkspaceAccessContext | None,
    ) -> list[MemoryAtom]:
        return await self._request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            list(aliases),
            identity_scope=scope,
            access=access,
        )

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        access: WorkspaceAccessContext | None,
    ) -> list[MemoryAtom]:
        return await self._request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request,
            access=access,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        scope: IdentityScope,
        access: WorkspaceAccessContext | None,
    ) -> ResolvedAgentProfile:
        return await self._request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            agent_alias,
            identity_scope=scope,
            access=access,
        )

    async def _request(self, route: str, *args: Any, **kwargs: Any) -> Any:
        """请求 backing 路由；路由未挂载即库不可达，显式失败。"""
        if route not in self._bus.list_routes():
            raise ResourceUnavailableError(
                "Patchouli backing 路由不可达",
                details={"reason": "backing_unreachable", "route": route},
            )
        return await self._bus.request(route, *args, **kwargs)


__all__ = ["BusCanonicalReadBackend"]
