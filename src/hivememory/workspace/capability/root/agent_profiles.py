"""root Agent Profile 管理 API：授权后创建与列出 Profile 源原子。

管理读取直接调用 Patchouli，不使用 Profile 解析缓存；创建时的 system
来源仅用于 provenance 展示，不成为资源可见性依据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from hivememory.core.errors import InvalidMemoryFieldError
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


class RootAgentProfileService:
    """只向保留的 system actor 提供的资源管理入口，不使用 actor 读取缓存。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer

    async def create_agent_profile(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        title: str,
        alias: str,
        summary: str = "",
        content: str = "",
        tags: list[str],
        agent_config: dict[str, Any] | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """在目标 Workspace 中创建 Agent Profile（root 管理用例）。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        # 只包装调用方提交字段的构造：输入不合法是 422，不是程序错误。
        try:
            index = IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType.AGENT_PROFILE,
                alias=alias,
            )
            payload = PayloadLayer(content=content, agent_config=agent_config)
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
            PatchouliRoutes.AGENT_PROFILE_CREATE,
            scope,
            atom,
        )

    async def list_agent_profiles(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        limit: int = 100,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """在目标 Workspace 中列出 Agent Profile（root 管理用例）。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_LIST,
            identity_scope=scope,
            limit=limit,
        )
