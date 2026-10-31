"""Import Bus 应用服务：把一次导入或落位请求翻译为导入授权与一次编排调用。

server 与 MCP 入口以 (user, 具体 Agent) 声明经统一认证网关取得请求级
context；本服务以导入授权把它兑现为本次请求内有效的 ``IdentityScope`` 与
认证通过的 principal 标识，再交给编排器。导入授权只校验准入与目标、不检查
actor 操作白名单：导入提交的是已经发生的交互，落位只为这一轮选定话题，两者
都不调用 actor 操作；落位之后的写入与更新仍由 actor API 逐次授权。
"""

from __future__ import annotations

from typing import Any

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models import WorkspaceIdentity
from hivememory.system.services.import_bus import (
    ImportBusIngestor,
    ImportedTurn,
    ImportTurnResult,
    InFlightTurn,
)
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class ImportBusService:
    """外部 harness 交互导入与写入时落位的入口服务，不持有跨请求的业务状态。"""

    def __init__(
        self,
        *,
        ingestor: ImportBusIngestor,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._ingestor = ingestor
        self._authorizer = operation_authorizer

    async def import_turn(
        self,
        turn: ImportedTurn,
        *,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> ImportTurnResult:
        """导入一轮完整记录，返回接纳、重复或跳过。

        ``access`` 是本次请求的请求级 context，``target_workspace`` 是这次
        导入的目标；授权失败按访问错误原样传播。
        """
        authorization = self._authorizer.authorize_import(access, target_workspace)
        return await self._ingestor.ingest(
            turn,
            identity_scope=authorization.identity_scope,
            principal_id=authorization.principal_id,
        )

    async def place_turn(
        self,
        turn: InFlightTurn,
        *,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> str:
        """为仍在进行的一轮落位，返回落点话题的真实 id。

        写入与更新在交互进行中提交，以落点话题派发；同一轮再次落位返回同一
        落点，这一轮结束导入时沿用它。``access`` 与 ``target_workspace`` 的
        含义与导入相同；授权失败按访问错误原样传播。
        """
        authorization = self._authorizer.authorize_import(access, target_workspace)
        return await self._ingestor.place(
            turn,
            identity_scope=authorization.identity_scope,
            principal_id=authorization.principal_id,
        )

    async def shutdown(self) -> dict[str, Any]:
        """停止接收并等待在途导入结束，返回等待的结果。"""
        return await self._ingestor.shutdown()


__all__ = ["ImportBusService"]
