"""Import Bus 应用服务：把一次导入或确定话题请求翻译为 root 授权与一次编排调用。

server 与 MCP 入口以请求声明的 principal、凭据与 (user, 具体 Agent) 经统一
认证网关认证，得到请求的工作 actor：它证明请求来自哪个调用来源、参与者是
哪个 Agent。导入与确定话题都是系统级操作，Agent 不能拥有：本服务再以系统自身
的 principal 认证同一用户在参与者驻留 Workspace 的 ``system``，以 root 授权
得到本次请求内有效的 ``IdentityScope``，连同参与的 Agent 与 principal 标识
交给编排器。system 驻留在参与者的 Workspace，root 授权因此同时确认目标就是
参与者驻留的 Workspace。actor 操作白名单不参与，空白名单的 Agent 也能导入；
确定话题之后的写入与更新仍由 actor API 按请求的工作 actor 逐次授权。
"""

from __future__ import annotations

from typing import Any

from hivememory.core.access import CallerPrincipal, WorkingActor
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.system.services.import_bus import (
    ImportBusIngestor,
    ImportedTurn,
    ImportTurnResult,
    InFlightTurn,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class ImportBusService:
    """外部 harness 交互导入与写入时确定话题的入口服务，不持有跨请求的业务状态。

    ``system_principal`` 与 ``system_adapter`` 是组合根注入的系统自身调用来源，
    与任务进程的注册入口相同。
    """

    def __init__(
        self,
        *,
        ingestor: ImportBusIngestor,
        access_gateway: ActorAuthenticationGateway,
        operation_authorizer: WorkspaceOperationAuthorizer,
        system_principal: CallerPrincipal,
        system_adapter: str,
    ) -> None:
        self._ingestor = ingestor
        self._access_gateway = access_gateway
        self._authorizer = operation_authorizer
        self._system_principal = system_principal
        self._system_adapter = system_adapter

    async def import_turn(
        self,
        turn: ImportedTurn,
        *,
        actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
    ) -> ImportTurnResult:
        """导入一轮完整记录，返回接纳、重复或跳过。

        ``actor`` 是入口为本次请求认证得到的工作 actor，``target_workspace``
        是这次导入的目标；认证与授权失败按访问错误原样传播。
        """
        system_actor, scope = await self._authorize_system(actor, target_workspace)
        return await self._ingestor.ingest(
            turn,
            system_actor=system_actor,
            identity_scope=scope,
            participant=actor.actor,
            principal_id=actor.principal.principal_id,
        )

    async def place_turn(
        self,
        turn: InFlightTurn,
        *,
        actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
    ) -> str:
        """为仍在进行的一轮确定话题，返回该话题的真实 id。

        写入与更新在交互进行中提交，以该话题派发；同一轮再次请求返回同一
        话题，这一轮结束导入时沿用它。``actor`` 与 ``target_workspace`` 的
        含义与导入相同；认证与授权失败按访问错误原样传播。
        """
        _system_actor, scope = await self._authorize_system(actor, target_workspace)
        return await self._ingestor.place(
            turn,
            identity_scope=scope,
            participant=actor.actor,
            principal_id=actor.principal.principal_id,
        )

    async def shutdown(self) -> dict[str, Any]:
        """停止接收并等待在途导入结束，返回等待的结果。"""
        return await self._ingestor.shutdown()

    async def _authorize_system(
        self, actor: WorkingActor, target_workspace: WorkspaceIdentity
    ) -> tuple[WorkingActor, IdentityScope]:
        """以系统自身的 principal 认证参与者驻留 Workspace 的 ``system``，再对目标执行 root 授权。

        目标不是参与者驻留的 Workspace 时，root 授权以
        ``target_workspace_not_resident`` 拒绝。系统自身的 principal 未登记时以
        ``unknown_principal`` 拒绝；认证网关关闭后同样拒绝，导入入口据此返回
        可重试的 503。
        """
        system_actor = await self._access_gateway.authenticate(
            adapter=self._system_adapter,
            principal=self._system_principal,
            credential=None,
            actor=ActorIdentity(user_id=actor.actor.user_id, agent_id=SYSTEM_AGENT_ID),
            workspace=actor.workspace,
        )
        return system_actor, self._authorizer.authorize_root(system_actor, target_workspace)


__all__ = ["ImportBusService"]
