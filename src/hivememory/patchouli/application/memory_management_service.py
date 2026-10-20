from __future__ import annotations

from typing import Any
from uuid import UUID

from hivememory.core.errors import (
    InvalidMemoryFieldError,
    ResourceUnavailableError,
)
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    MemoryType,
    require_identity_scope,
)
from hivememory.core.models.query import QueryFilters
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_management import MemoryCreation, MemoryEdit
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.utils.uuid import normalize_uuid


class MemoryManagementService:
    """Patchouli 面向公开记忆管理/读取 API 的应用服务。

    管理用例由 workspace root API 授权，actor 读取由 actor 能力授权点
    校验；本层是授权点以下的资源 owner，只接收组装后的 ``IdentityScope``，
    不接收访问 context。资源归属与 ``MemoryAccessPolicy`` 仍由本层与存储
    边界独立校验，读取 backing 路由同时服务于阶段编排与 actor 读取视图。
    """

    def __init__(self, *, bus: PatchouliBus) -> None:
        self._bus = bus

    # ---- 管理用例（root API） ----

    async def create_memory(
        self,
        identity_scope: IdentityScope,
        creation: MemoryCreation,
    ) -> MemoryAtom:
        """以提交的材料在目标 Workspace 中创建记忆，来源记录取自本次发起者。"""
        scope = require_identity_scope(identity_scope)
        if creation.memory_type is MemoryType.AGENT_PROFILE:
            # Profile 只经 Profile 创建入口创建，记忆入口不能借道绕开 Profile 的约束。
            raise InvalidMemoryFieldError("AGENT_PROFILE 只能经 Agent Profile 创建入口创建")
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            scope.workspace_identity,
            from_actor=scope.actor_identity,
            memory_type=creation.memory_type,
            title=creation.title,
            summary=creation.summary,
            content=creation.content,
            tags=list(creation.tags),
            alias=creation.alias,
        )

    async def list_memories(
        self,
        *,
        identity_scope: IdentityScope,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 20,
        exclude_types: list[str] | None = None,
        refresh_vitality: bool = True,
    ) -> list[MemoryAtom]:
        scope = require_identity_scope(identity_scope)
        excluded = set(exclude_types or [])
        atoms = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            query=query,
            filters=filters,
            limit=limit,
            # owner 管理读取：ownership hard boundary 之内
            # 读取该 Workspace 全部 Memory，不执行 Agent 可见性过滤；
            # 该语义只供经 root API 授权的管理读取使用。
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
        identity_scope: IdentityScope,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        scope = require_identity_scope(identity_scope)
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            # owner 管理读取：归属硬边界内不按 Agent 可见性过滤，同 list_memories。
            enforce_actor_visibility=False,
        )
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def update_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        edit: MemoryEdit,
    ) -> MemoryAtom | None:
        """管理操作已在能力层授权；内部编辑只按目标归属定位记忆。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_UPDATE,
            normalize_uuid(memory_id),
            belong_to=scope.workspace_identity,
            title=edit.title,
            summary=edit.summary,
            content=edit.content,
            alias=edit.alias,
            tags=list(edit.tags) if edit.tags is not None else None,
            agent_config=edit.agent_config,
        )

    async def delete_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
    ) -> bool:
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_DELETE,
            scope.workspace_identity,
            normalize_uuid(memory_id),
        )

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        positive: bool,
        source: str,
    ) -> Any:
        scope = require_identity_scope(identity_scope)
        # 接线缺失在调用前判定；已注册 handler 的内部错误必须原样传播。
        if PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK not in self._bus.list_routes():
            raise ResourceUnavailableError(details={"reason": "lifecycle_unavailable"})
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            normalize_uuid(memory_id),
            belong_to=scope.workspace_identity,
            positive=positive,
            source=source,
        )

    # ---- Actor 可见读取 backing（operation 授权在 workspace 能力层） ----

    async def read_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        """Actor-visible 的 canonical UUID 点读 backing，读取视图按本次授权 scope 交付。

        与管理 GET 的区别：可见性由 Patchouli 按 MemoryAccessPolicy 强制
        （enforce=True），不可见与缺失统一返回 ``None``。
        """
        scope = require_identity_scope(identity_scope)
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            enforce_actor_visibility=True,
        )
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[MemoryAtom]:
        """语义检索 backing（能力层授权 ``memory.search``），返回完整原子列表。

        检索请求自带 ``identity_scope``（授权点组装后冻结在请求内），不
        接收单独的 scope 参数。
        """
        scope = require_identity_scope(request.identity_scope)
        # 公开输入只在边界持有 scope，本地路由与引擎接收独立的归属和发起者。
        query = RetrievalQuery(
            semantic_query=request.semantic_query,
            keywords=request.keywords or [],
            filters=request.filters or QueryFilters(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            query,
            top_k=request.top_k,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        identity_scope: IdentityScope,
    ) -> list[MemoryAtom]:
        """alias 批量读取 backing，按 READ/RUN/UPDATE 本次授权 scope 返回，只含实际可读的完整原子。"""
        scope = require_identity_scope(identity_scope)
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            aliases,
            scope.workspace_identity,
            from_actor=scope.actor_identity,
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
