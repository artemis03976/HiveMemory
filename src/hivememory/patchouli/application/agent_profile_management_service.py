from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    MemoryType,
    ResolvedAgentProfile,
)
from hivememory.patchouli.application.access_consumption import backing_scope, verified_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext, WorkspaceAccessVerifier


class AgentProfileManagementService:
    """Patchouli 面向公开 agent profile 管理/读取 API 的应用服务。

    Profile 管理写入/列表沿用 AGENT_PROFILE atom 的既有绑定例外，其
    ``management.memory`` 行为授权已上移到 workspace 能力层（A1 访问边界
    返工第 4.3 节）；本层只经 :func:`verified_scope` 校验 access 有效性与
    DTO scope 一致性，缺少 access 一律拒绝。

    ``get_agent_profile`` 是 Profile 读取的 L2 backing（A2 §2.3/§8 D-3）：
    ``profile.read`` 的行为授权由 workspace 能力层在 backing 调用前执行，
    此处只校验 access 有效性；返回 ``ResolvedAgentProfile``（AgentProfile +
    源原子 policy 依据与关联），可见性校验在库内独立成立。它是迁移期受信
    适配清单中的方法：Alice 的 profile resolver 仍不带 access 调用，裸
    scope 适配保留（见 ``access_consumption``）。
    """

    def __init__(self, *, bus: Any, access_guard: WorkspaceAccessVerifier) -> None:
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
    ) -> ResolvedAgentProfile:
        """Profile 定义解析 backing：唯一解析规则 + 源原子 policy 依据与关联。"""
        scope = backing_scope(
            access,
            identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
            identity_scope=scope,
        )
