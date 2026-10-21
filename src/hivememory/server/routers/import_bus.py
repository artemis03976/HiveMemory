"""Import Bus 路由 — POST /api/v1/import/turns"""

from fastapi import APIRouter, Depends, Request

from hivememory.server.deps import (
    RequestIdentitySelection,
    get_access_gateway,
    get_identity_selection,
    get_import_bus_service,
    get_import_request_limit,
    get_server_principal_id,
    request_access_for_claims,
    resolve_request_identity_claims,
)
from hivememory.server.error_handlers import APIError, error_responses
from hivememory.server.models.import_bus import ImportTurnRequest, ImportTurnResponse
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.workspace.authentication import ActorAuthenticationGateway

router = APIRouter(
    responses=error_responses(400, 403, 404, 413, 422, 500, 503),
    tags=["import"],
)


async def enforce_import_body_limit(
    request: Request,
    limit: int = Depends(get_import_request_limit),
) -> None:
    """请求体超过 ``import_bus.max_request_bytes`` 时以 413 拒绝。

    过大的轮次重发也不会变小，因此是不可重试的拒绝：连接器记录后继续
    处理后续轮次。请求体由框架读取并缓存，这里只按实际字节数判断。
    """
    if len(await request.body()) > limit:
        raise APIError("request.body_too_large")


@router.post(
    "/import/turns",
    response_model=ImportTurnResponse,
    dependencies=[Depends(enforce_import_body_limit)],
)
async def import_turn(
    body: ImportTurnRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: ImportBusService = Depends(get_import_bus_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> ImportTurnResponse:
    """导入 harness 一轮已结束交互的完整记录，不返回记忆上下文。

    导入是 Agent action：body 必须携带连接器所代表的具体 ``agent_id``，
    用户与 Workspace 选择只来自统一请求头。server 以 (user, 具体 Agent)
    声明、自身 principal 与 ``http`` adapter 经统一认证网关取得请求级
    context，请求结束时失效；认证前不组装 ``IdentityScope``。

    状态码区分是否可重试：校验失败、Agent 缺失或为 ``system``、请求体过大
    与准入失败不可重试；认证网关关闭、Gateway 暂不可用与提交队列拥塞
    返回 503，连接器保留该轮稍后重发。
    """
    claims = resolve_request_identity_claims(selection, require_agent=True, agent_id=body.agent_id)
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as request_access:
        result = await service.import_turn(
            body.to_imported_turn(),
            access=request_access.access,
            target_workspace=request_access.target_workspace,
        )
    return ImportTurnResponse(status=result.status, interaction_id=result.interaction_id)
