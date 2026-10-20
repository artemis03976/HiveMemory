"""workspace 读取视图经总线访问 Patchouli backing 读取路由。

实现 ``workspace.resolution.CanonicalReadBackend``：resolver 的 L2 冷读经
``GlobalSystemBus`` 调用 Patchouli 公共 backing 路由（第二层 client-server），
不形成递归，也不持有 Patchouli 的 Runtime、Service 或存储对象。它不是 API，
由组合根注入 ``WorkspaceReadView``；actor API 的读取与任务进程阶段调用中的
Profile 解析都经读取视图到达这里。

放在 ``capability`` 而不是 ``resolution``：读取本体按导入边界只依赖 core 与
workspace 自身，经本协议冷读，不导入总线；依赖总线机制的适配器留在可以
依赖 ``components`` 的子包中。

``scope`` 是授权点（actor API 的操作授权或任务进程的阶段授权）组装的可信
坐标，原样传给 Patchouli backing；本模块位于授权点以下，不接收访问 context。

库不可达（backing 路由未挂载，如 Patchouli 未启动或已卸载）时抛出
``ResourceUnavailableError``，由调用方显式失败，不以空结果伪装成功；
存储错误与领域错误按原结构化错误传播。
"""

from __future__ import annotations

from uuid import UUID

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.route import Route
from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import IdentityScope, MemoryAtom, ResolvedAgentProfile
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.contracts import PatchouliRoutes


class BusCanonicalReadBackend:
    """基于 ``GlobalSystemBus`` 的 Patchouli canonical/Profile 读取 backing。"""

    def __init__(self, global_bus: GlobalSystemBus) -> None:
        self._bus = global_bus

    async def read(
        self,
        memory_id: UUID,
        *,
        scope: IdentityScope,
    ) -> MemoryAtom | None:
        return await self._request(
            PatchouliRoutes.MEMORY_READ,
            memory_id,
            identity_scope=scope,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
    ) -> list[MemoryAtom]:
        return await self._request(
            PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            list(aliases),
            identity_scope=scope,
        )

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[MemoryAtom]:
        return await self._request(
            PatchouliRoutes.MEMORY_RETRIEVE,
            request,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        scope: IdentityScope,
    ) -> ResolvedAgentProfile:
        return await self._request(
            PatchouliRoutes.GET_AGENT_PROFILE,
            agent_alias,
            identity_scope=scope,
        )

    async def _request[**P, R](self, route: Route[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """请求 backing 路由；路由未挂载即库不可达，显式失败。"""
        if route not in self._bus.list_routes():
            raise ResourceUnavailableError(
                "Patchouli backing 路由不可达",
                details={"reason": "backing_unreachable", "route": route},
            )
        return await self._bus.request(route, *args, **kwargs)


__all__ = ["BusCanonicalReadBackend"]
