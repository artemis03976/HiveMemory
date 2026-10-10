"""actor API：逐次检查操作权限，读取视图交付与写入意图登记属于 workspace。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import AgentProfile, WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.profile import ProfileResolver


class AgentApplicationService:
    """actor 资源入口，访问 context 只由本层授权并转为本次操作 scope。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        profile_reader: ProfileResolver,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._profile_reader = profile_reader

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """Profile 定义读取（``profile.read``）：交付 AgentProfile 独立副本。

        未指定 alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；
        自定义 alias 缺失、不可见、类型不符或配置损坏均显式失败，不降级为
        默认配置。读取 Profile 不等于获得其描述的权限。
        """
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.PROFILE_READ, target_workspace
        )
        return await self._profile_reader.get(agent_alias, scope=scope)
