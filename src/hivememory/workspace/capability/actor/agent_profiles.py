"""actor CALL API：以 ``agent.call`` 授权后经 Profile 读取器解析被调用方的图纸。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.workspace.contracts.operations import DeriveCallThreadRequest

if TYPE_CHECKING:
    from hivememory.core.access import WorkingActor
    from hivememory.core.models.agent import AgentProfile
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.profile import ProfileResolver


class ActorAgentService:
    """actor 资源入口：在入口以工作 actor 授权 ``agent.call``，再读取全局的图纸。"""

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
        actor: WorkingActor,
    ) -> AgentProfile:
        """CALL 的 actor 操作（``agent.call``）：确定被调用方的图纸。

        ``actor`` 是调用方线程的工作 actor，授权目标取它的驻留 Workspace。只做权限检查
        与图纸解析，图纸是全局的，结果只有图纸、不含身份；被调用方在 Workspace 中的准入
        与子线程的派生是线程的生命周期动作，由操作入口在同一次提交内交给线程表。未指定
        alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；自定义 alias
        缺失、文件不合格或已停用均显式失败，不降级为默认配置。
        """
        self._authorizer.authorize_agent(actor, WorkspaceOperation.AGENT_CALL, actor.workspace)
        return await self._profile_reader.get(request.agent_alias)
