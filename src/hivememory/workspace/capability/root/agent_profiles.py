"""root Agent Profile 管理 API：授权后创建与列出 Profile 源原子。

管理读取直接调用 Patchouli，不使用 Profile 解析缓存；创建时的 system
来源仅用于 provenance 展示，不成为资源可见性依据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.models import (
    MemoryAtom,
)
from hivememory.patchouli.contracts import AgentProfileCreation, PatchouliRoutes

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
        """在目标 Workspace 中创建 Agent Profile（root 管理用例）。

        只提交创建材料，原子由 Patchouli 构造；字段非法时 Patchouli 以
        ``InvalidMemoryFieldError`` 拒绝。
        """
        scope = self._authorizer.authorize_root(access, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_CREATE,
            identity_scope=scope,
            creation=AgentProfileCreation(
                title=title,
                alias=alias,
                summary=summary,
                content=content,
                tags=tuple(tags),
                agent_config=agent_config,
            ),
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
