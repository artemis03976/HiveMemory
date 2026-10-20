"""actor CALL API：操作授权后经 Profile 读取视图解析目标图纸，保留资源可见性。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.workspace.contracts.operations import DeriveCallThreadRequest

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models.agent import AgentProfile
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
        request: DeriveCallThreadRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """CALL 的 actor 操作（``agent.call``）：以调用方可见的图纸确定被调用方。

        只做权限检查与图纸解析，结果只有图纸、不含凭据；子线程的派生与结束是
        线程的生命周期动作，由操作入口在同一次提交内交给线程表。未指定 alias
        或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；自定义 alias
        缺失、不可见、类型不符或配置损坏均显式失败，不降级为默认配置。
        """
        scope = self._authorizer.authorize_thread_operation(access, WorkspaceOperation.AGENT_CALL)
        return await self._profile_reader.get(request.agent_alias, scope=scope)
