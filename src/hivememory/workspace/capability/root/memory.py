"""root 记忆管理 API：授权后直接调用 Patchouli，不进入 actor 读取视图。

管理读取沿用 owner-management 语义：保留 Workspace 归属硬边界，
资源 owner 不按 Agent 的 PUBLIC/PRIVATE/TEAM 策略裁剪管理结果。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from hivememory.core.errors import (
    InvalidMemoryFieldError,
    ResourceNotFoundError,
)
from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.utils.time import utc_now

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootMemoryService:
    """只向保留的 system actor 提供的资源管理入口，不使用 actor 读取缓存。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer

    async def create_memory(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理创建入口（root 授权）：在目标 Workspace 中创建 Memory。

        ``provenance.source_agent_id`` 记录来源 actor（管理入口为保留
        ``system``），只作 provenance 展示，不参与可见性授权。
        """
        scope = self._authorizer.authorize_root(access, target_workspace)
        # 只包装调用方提交字段的构造：输入不合法是 422，不是程序错误。
        try:
            index = IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType(memory_type),
                alias=alias,
            )
            payload = PayloadLayer(content=content)
        except ValidationError as exc:
            raise InvalidMemoryFieldError.from_validation_error(exc) from exc
        # 创建时点在提交边界取一次 now，created/updated/decay anchor 同值，
        # 统一由 Patchouli 完整写入路径赋值。
        now = utc_now()
        atom = MemoryAtom(
            meta=MetaData(
                workspace_identity=scope.workspace_identity,
                provenance=MemoryProvenance(
                    source_agent_id=scope.actor_identity.agent_id,
                    source_team_id=scope.actor_identity.team_id,
                ),
                access_policy=MemoryAccessPolicy.public(),
                created_at=now,
                updated_at=now,
                lifecycle=MemoryLifecycleState(decay_anchor_at=now),
            ),
            index=index,
            payload=payload,
        )
        return await self._global_bus.request(
            PatchouliRoutes.MEMORY_CREATE,
            scope,
            atom,
        )

    async def list_memories(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """管理读取入口（root 授权）：在目标 Workspace 中列出 Memory。

        按 owner-management 语义返回该 Workspace 的全部 Memory（不含
        Agent Profile），不做 Agent ``MemoryAccessPolicy`` 过滤。
        """
        scope = self._authorizer.authorize_root(access, target_workspace)
        filters = self._build_filters(memory_type=memory_type)
        return await self._global_bus.request(
            PatchouliRoutes.MEMORY_LIST,
            identity_scope=scope,
            query=query,
            filters=filters if filters else None,
            limit=limit,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
            refresh_vitality=True,
        )

    async def get_memory(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理读取入口（root 授权）：在目标 Workspace 中读取 Memory。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        atom = await self._global_bus.request(
            PatchouliRoutes.MEMORY_GET,
            memory_id,
            identity_scope=scope,
            refresh_vitality=True,
        )
        if atom is None:
            raise ResourceNotFoundError("记忆不存在")
        return atom

    async def update_memory(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理更新入口（root 授权）：显式授权 mutation，且不改变原 ownership/provenance。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        atom = await self._global_bus.request(
            PatchouliRoutes.MEMORY_UPDATE,
            memory_id,
            identity_scope=scope,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )
        if atom is None:
            raise ResourceNotFoundError("记忆不存在")
        return atom

    async def record_feedback(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        positive: bool,
        source: str,
        access: WorkspaceAccessContext,
    ):
        """管理反馈入口（root 授权）：在目标 Workspace 中记录反馈。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
            memory_id,
            identity_scope=scope,
            positive=positive,
            source=source,
        )

    async def delete_memory(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> bool:
        """管理删除入口（root 授权）：在目标 Workspace 中删除 Memory。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.MEMORY_DELETE,
            memory_id,
            identity_scope=scope,
        )

    @staticmethod
    def _build_filters(
        *,
        memory_type: str | None,
    ) -> dict[str, str]:
        filters = {}
        if memory_type:
            filters["index.memory_type"] = memory_type
        return filters
