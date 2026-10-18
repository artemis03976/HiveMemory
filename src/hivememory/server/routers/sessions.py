"""会话管理 HTTP 适配层；身份解析与授权沿用统一 root 入口。"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from hivememory.server.deps import (
    RequestAccess,
    RequestIdentitySelection,
    get_access_gateway,
    get_identity_selection,
    get_root_session_service,
    get_server_principal_id,
    request_access_for_claims,
    resolve_request_identity_claims,
)
from hivememory.server.error_handlers import error_responses
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.sessions import ConversationSession, SessionSummary

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["sessions"])


class SessionTitleRequest(BaseModel):
    """创建或重命名会话的标题。"""

    title: str = Field(default="", max_length=200)


async def session_access(
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> AsyncIterator[RequestAccess]:
    """每次管理请求独立认证，响应完成后撤销 context。"""
    claims = resolve_request_identity_claims(selection)
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as access:
        yield access


@router.post("/sessions", response_model=ConversationSession, status_code=201)
async def create_session(
    body: SessionTitleRequest,
    access: RequestAccess = Depends(session_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSession:
    """创建会话；空标题在首轮写入时补全。"""
    return service.create_session(
        title=body.title, target_workspace=access.target_workspace, access=access.access
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    access: RequestAccess = Depends(session_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> tuple[SessionSummary, ...]:
    """列出当前 Workspace 的会话。"""
    return service.list_sessions(target_workspace=access.target_workspace, access=access.access)


@router.get("/sessions/{session_id}", response_model=ConversationSession)
async def get_session(
    session_id: str,
    access: RequestAccess = Depends(session_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSession:
    """读取会话全部轮次。"""
    return service.get_session(
        session_id, target_workspace=access.target_workspace, access=access.access
    )


@router.patch("/sessions/{session_id}", response_model=ConversationSession)
async def rename_session(
    session_id: str,
    body: SessionTitleRequest,
    access: RequestAccess = Depends(session_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSession:
    """重命名会话。"""
    return service.rename_session(
        session_id, title=body.title, target_workspace=access.target_workspace, access=access.access
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    access: RequestAccess = Depends(session_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> Response:
    """删除空闲会话。"""
    service.delete_session(
        session_id, target_workspace=access.target_workspace, access=access.access
    )
    return Response(status_code=204)
