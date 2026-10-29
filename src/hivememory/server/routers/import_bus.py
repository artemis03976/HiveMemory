"""Import Bus 路由 — POST /api/v1/import/turns"""

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.routing import APIRoute

from hivememory.server.deps import (
    HTTP_ADAPTER,
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


class BodyLimitedRoute(APIRoute):
    """在框架读取并解析请求体之前，按 ``import_bus.max_request_bytes`` 拒绝过大的请求。

    声明的 Content-Length 超过上限时不读取请求体；没有声明长度时边读边计数，
    超过上限即停止读取。两种情况都以 413 拒绝：过大的轮次重发也不会变小，
    是不可重试的拒绝，连接器记录后继续处理后续轮次。读完的请求体交还给
    框架，不再读取第二次。
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def limited_handler(request: Request) -> Response:
            limit_provider = request.app.dependency_overrides.get(
                get_import_request_limit, get_import_request_limit
            )
            limit = limit_provider()
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                raise APIError("request.body_too_large")
            received = bytearray()
            async for chunk in request.stream():
                received.extend(chunk)
                if len(received) > limit:
                    raise APIError("request.body_too_large")
            request._body = bytes(received)
            return await handler(request)

        return limited_handler


router = APIRouter(
    responses=error_responses(400, 403, 404, 413, 422, 500, 503),
    tags=["import"],
    route_class=BodyLimitedRoute,
)


@router.post(
    "/import/turns",
    response_model=ImportTurnResponse,
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

    确认表示交互已经应用。状态码区分是否可重试：校验失败、Agent 缺失或为
    ``system``、请求体过大与准入失败不可重试；认证网关关闭、Gateway 暂不
    可用、提交队列拥塞与等待应用超时返回 503，连接器保留该轮稍后重发；
    应用失败返回 500，连接器有限次重发，重发时重新路由并再次提交。
    """
    claims = resolve_request_identity_claims(selection, require_agent=True, agent_id=body.agent_id)
    async with request_access_for_claims(
        claims,
        gateway=gateway,
        principal_id=principal_id,
        adapter=HTTP_ADAPTER,
        credential=None,
    ) as request_access:
        result = await service.import_turn(
            body.to_imported_turn(),
            access=request_access.access,
            target_workspace=request_access.target_workspace,
        )
    return ImportTurnResponse(status=result.status, interaction_id=result.interaction_id)
