from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    MemoryType,
)
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.patchouli.application.access_consumption import (
    required_scope,
    verified_scope,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.utils.uuid import normalize_uuid
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class MemoryManagementService:
    """Patchouli 面向公开记忆管理/读取 API 的应用服务。

    每个用例绑定明确的 operation（A1 计划第 4.1 节绑定基线），在资源
    读取或副作用之前经 ``WorkspaceAccessGuard`` 执行共享行为检查：

    - 管理 CRUD/GET/LIST/feedback：``management.memory``——owner-management
      读取语义只由该 operation 授权，Agent 的 ``resource.read``/
      ``resource.search`` 调用同一管理入口会在校验层失败；
    - ``read_memory``（Actor-visible UUID 点读）：``resource.read``；
    - ``retrieve``（语义检索）：``resource.search``；
    - ``retrieve_by_aliases``（正式 alias 读取）：``resource.read``。

    迁移期兼容（A1 第 6 节冻结清单）：未提供 ``access`` 的旧调用方（管理
    入口 HTTP 链路与 Alice resolver 代理路由）按受信适配走裸
    ``IdentityScope``；该分支不得保留无许可的公共成功语义，A6 完成生产
    消费者切换后删除。
    """

    def __init__(
        self,
        *,
        bus,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        self._bus = bus
        self._access_guard = access_guard

    # ---- 管理用例（management.memory） ----

    async def create_memory(
        self,
        identity_scope: IdentityScope | None = None,
        atom: MemoryAtom | None = None,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        if atom is None:
            raise ValueError("create_memory 需要 atom 载荷")
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        if atom.workspace_identity != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            scope,
            atom,
        )

    async def list_memories(
        self,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 20,
        exclude_types: list[str] | None = None,
        refresh_vitality: bool = True,
    ) -> list[MemoryAtom]:
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        excluded = set(exclude_types or [])
        atoms = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            identity_scope=scope,
            query=query,
            filters=filters,
            limit=limit,
            # owner-management 语义（D4）：ownership hard boundary 之内
            # 读取该 Workspace 全部 Memory，不执行 Agent 可见性过滤；
            # 该语义只由 management.memory 授权。
            enforce_actor_visibility=False,
        )
        atoms = [
            atom
            for atom in atoms
            if self._memory_type_value(atom.index.memory_type) not in excluded
        ]
        if refresh_vitality:
            await self._refresh_vitality_for_response(atoms)
        return atoms

    async def get_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
            identity_scope=scope,
            # owner-management 语义（D4），同 list_memories。
            enforce_actor_visibility=False,
        )
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def update_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom | None:
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_UPDATE,
            normalize_uuid(memory_id),
            identity_scope=scope,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )

    async def delete_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
    ) -> bool:
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_DELETE,
            scope,
            normalize_uuid(memory_id),
        )

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope | None = None,
        access: WorkspaceAccessContext | None = None,
        positive: bool,
        source: str,
    ):
        scope = verified_scope(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            normalize_uuid(memory_id),
            identity_scope=scope,
            positive=positive,
            source=source,
        )

    # ---- Actor 可见读取用例（resource.read / resource.search） ----

    async def read_memory(
        self,
        memory_id: UUID | str,
        *,
        access: WorkspaceAccessContext | None = None,
        identity_scope: IdentityScope | None = None,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        """Actor-visible 的 canonical UUID 点读（resource.read）。

        与管理 GET 的区别：可见性由 Patchouli 按 MemoryAccessPolicy 强制
        （enforce=True），不可见与缺失统一返回 ``None``；本用例不在迁移
        兼容清单内——缺少 access 一律拒绝。
        """
        scope = required_scope(
            access, WorkspaceOperation.RESOURCE_READ, identity_scope,
            access_guard=self._access_guard,
        )
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
            identity_scope=scope,
            enforce_actor_visibility=True,
        )
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> RetrievalResponse:
        # 语义检索按 resource.search 授权；检索请求中的 scope 不得偏离
        # access 上下文（迁移期无 access 的调用走受信适配）。
        verified_scope(
            access, WorkspaceOperation.RESOURCE_SEARCH, request.identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            request,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        identity_scope: IdentityScope | None = None,
        *,
        access: WorkspaceAccessContext | None = None,
    ) -> RetrievalResponse:
        scope = verified_scope(
            access, WorkspaceOperation.RESOURCE_READ, identity_scope,
            access_guard=self._access_guard,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            aliases,
            scope,
        )

    @staticmethod
    def _memory_type_value(memory_type: MemoryType | str) -> str:
        return memory_type.value if hasattr(memory_type, "value") else str(memory_type)

    async def _refresh_vitality_for_response(self, atoms: list[MemoryAtom]) -> None:
        if not atoms:
            return
        try:
            await self._bus.request(
                PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
                atoms,
                persist=False,
            )
        except Exception:
            return
