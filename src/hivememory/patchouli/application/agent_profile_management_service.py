from __future__ import annotations

from typing import Any

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    MemoryType,
    ResolvedAgentProfile,
    require_identity_scope,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class AgentProfileManagementService:
    """Patchouli 面向公开 agent profile 管理/读取 API 的应用服务。

    Profile 管理写入与列表由 workspace root API 授权；actor 读取与进程
    阶段读取分别在对应授权点校验。本层只接收组装后的 ``IdentityScope``，
    不接收访问 context。读取返回 ``ResolvedAgentProfile``（Profile、源原子
    policy 依据与关联），读取视图在交付时仍按 actor 检查可见性。
    """

    def __init__(self, *, bus: Any) -> None:
        self._bus = bus

    async def create_agent_profile(
        self,
        identity_scope: IdentityScope,
        atom: MemoryAtom | None = None,
    ) -> MemoryAtom:
        if atom is None:
            raise ValueError("create_agent_profile 需要 atom 载荷")
        scope = require_identity_scope(identity_scope)
        if atom.workspace_identity != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            scope.workspace_identity,
            atom,
        )
        return atom

    async def list_agent_profiles(
        self,
        *,
        identity_scope: IdentityScope,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> ResolvedAgentProfile:
        """Profile 定义解析 backing：唯一解析规则 + 源原子 policy 依据与关联。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )
