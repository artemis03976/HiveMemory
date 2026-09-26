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
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class AgentProfileManagementService:
    """Patchouli 面向公开 agent profile 管理/读取 API 的应用服务。

    Profile 管理写入/列表沿用 AGENT_PROFILE atom 的既有绑定例外，绑定
    ``management.memory``，在资源读取前经共享行为检查（A1 计划第 4.1 节）。

    ``get_agent_profile`` 是 Profile 读取的 L2 backing（A2 §2.3/§8 D-3）：
    ``profile.read`` 的行为授权由 workspace 能力层在 backing 调用前执行，
    此处只校验 access 有效性；返回 ``ResolvedAgentProfile``（AgentProfile +
    源原子 policy 依据与关联），可见性校验在库内独立成立。无 access 的
    既有调用方（Alice profile resolver、Patchouli prepare）走 A1 第 6 节
    兼容清单，A6 完成消费者切换后收紧。
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
