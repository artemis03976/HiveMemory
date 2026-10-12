"""actor CALL API：操作授权后经 Profile 读取视图解析目标，保留资源可见性。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.workspace.contracts.operations import ResolveCallTargetRequest

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import AgentProfile
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.profile import ProfileResolver


class ActorAgentService:
    """actor 资源入口，访问 context 只由本层授权并转为本次操作 scope。"""

    def __init__(
        self,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        profile_reader: ProfileResolver,
    ) -> None:
        self._authorizer = operation_authorizer
        self._profile_reader = profile_reader

    async def resolve_call_target(
        self,
        request: ResolveCallTargetRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """CALL 目标 Profile 读取（``agent.call``）：交付 AgentProfile 独立副本。

        未指定 alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；
        自定义 alias 缺失、不可见、类型不符或配置损坏均显式失败，不降级为
        默认配置。读取 Profile 不等于获得其描述的权限。
        """
        scope = self._authorizer.authorize_thread_operation(access, WorkspaceOperation.AGENT_CALL)
        return await self._profile_reader.get(request.agent_alias, scope=scope)
