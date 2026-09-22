from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    MemoryType,
    ProfileSnapshot,
)
from hivememory.patchouli.application.access_consumption import verified_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class AgentProfileManagementService:
    """Patchouli 面向公开 agent profile 管理/读取 API 的应用服务。

    Profile 读取用例绑定 ``profile.read`` operation，经局部
    ``GET_AGENT_PROFILE_SNAPSHOT`` 返回携带 source atom UUID/revision 的
    不可变快照；Profile 管理写入/列表沿用 AGENT_PROFILE atom 的既有绑定
    例外，绑定 ``management.memory``，与读取分别授权、互不推导（A1
    计划第 4.1 节）。管理/读取入口均在资源读取前经共享行为检查。

    ``get_agent_profile`` 保留为既有裸 Profile 契约的兼容投影（Alice
    resolver 消费，A1 第 6 节兼容清单），A6 完成消费者切换后收紧。
    """

    def __init__(self, *, bus: Any, access_guard: WorkspaceAccessGuard) -> None:
        self._bus = bus
        self._access_guard = access_guard

    async def create_agent_profile(
        self,
        identity_scope: IdentityScope | None = None,
        atom: MemoryAtom | None = None,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        if atom is None:
            raise ValueError("create_agent_profile 需要 atom 载荷")
        scope = verified_scope(
            access,
            WorkspaceOperation.MANAGEMENT_MEMORY,
            identity_scope,
            access_guard=self._access_guard,
        )
        if atom.workspace_identity != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            scope,
            atom,
        )
        return atom

    async def list_agent_profiles(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        scope = verified_scope(
            access,
            WorkspaceOperation.MANAGEMENT_MEMORY,
            identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            identity_scope=scope,
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
    ) -> AgentProfile:
        snapshot = await self.get_agent_profile_snapshot(
            agent_alias,
            identity_scope=identity_scope,
            access=access,
        )
        return snapshot.profile

    async def get_agent_profile_snapshot(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
    ) -> ProfileSnapshot:
        """读取 Profile 快照（profile.read）：唯一解析规则 + source 归属投影。"""
        scope = verified_scope(
            access,
            WorkspaceOperation.PROFILE_READ,
            identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE_SNAPSHOT,
            agent_alias,
            identity_scope=scope,
        )
