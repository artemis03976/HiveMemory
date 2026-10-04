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

    Profile 管理写入/列表沿用 AGENT_PROFILE atom 的既有绑定例外，其
    ``management.memory`` 行为授权、``get_agent_profile`` 的
    ``profile.read`` 行为授权都在 workspace 能力层完成（A1 访问边界返工
    第 4.5/4.6 节）；本层是授权点以下的资源 owner，只接收授权点组装的
    ``IdentityScope``，不接收访问 context。``get_agent_profile`` 返回
    ``ResolvedAgentProfile``（AgentProfile + 源原子 policy 依据与关联），
    可见性校验在库内独立成立；Alice 的 profile resolver 仍按原签名直接
    调用（无 operation 授权，总 Idea 15.5 的已知缺口）。
    """

    def __init__(self, *, bus: Any) -> None:
        self._bus = bus

    async def create_agent_profile(
        self,
        identity_scope: IdentityScope | None = None,
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
            scope,
            atom,
        )
        return atom

    async def list_agent_profiles(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        scope = require_identity_scope(identity_scope)
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
    ) -> ResolvedAgentProfile:
        """Profile 定义解析 backing：唯一解析规则 + 源原子 policy 依据与关联。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
            identity_scope=scope,
        )
