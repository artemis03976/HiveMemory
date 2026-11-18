"""root Agent Profile 管理 API：授权后新建、编辑、删除与列出 Profile。

Profile 是 Patchouli 的 AgentProfileStore 中的文件，全局而不属于任何 Workspace。这里以
请求驻留的 Workspace 做 root 授权，再转交 Patchouli 的公开路由，字段校验与读写都在那里。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from hivememory.patchouli.contracts import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import StoredAgentProfile, WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootAgentProfileService:
    """只向保留的 system actor 提供的 Agent Profile 管理入口。"""

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
        agent_id: str,
        fields: Mapping[str, Any],
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> StoredAgentProfile:
        """新建 Profile；同名已存在时以 ``AgentProfileExistsError`` 拒绝。"""
        self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_CREATE, agent_id=agent_id, fields=dict(fields)
        )

    async def update_agent_profile(
        self,
        agent_id: str,
        changes: Mapping[str, Any],
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> StoredAgentProfile:
        """修改 Profile 中给出的字段；不存在时抛出 ``ResourceNotFoundError``。"""
        self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_UPDATE, agent_id=agent_id, changes=dict(changes)
        )

    async def delete_agent_profile(
        self,
        agent_id: str,
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> bool:
        """删除 Profile，返回是否确有 Profile 被删除。"""
        self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_DELETE, agent_id=agent_id
        )

    async def list_agent_profiles(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> list[StoredAgentProfile]:
        """列出全部合格的 Profile，含停用的。"""
        self._authorizer.authorize_root(actor, target_workspace)
        return await self._global_bus.request(PatchouliRoutes.AGENT_PROFILE_LIST)
