"""会话管理 HTTP 适配层；身份解析与授权沿用统一 root 入口。"""

from fastapi import APIRouter, Depends, Response

from hivememory.server.deps import RequestAccess, get_request_access, get_root_session_service
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.session import (
    ConversationSessionResponse,
    SessionSummaryResponse,
    SessionTitleRequest,
)
from hivememory.workspace.capability.root.session import RootSessionService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["sessions"])


@router.post("/sessions", response_model=ConversationSessionResponse, status_code=201)
async def create_session(
    body: SessionTitleRequest,
    access: RequestAccess = Depends(get_request_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSessionResponse:
    """创建会话；空标题在首轮写入时补全。"""
    session = service.create_session(
        title=body.title, target_workspace=access.target_workspace, access=access.access
    )
    return ConversationSessionResponse.from_session(session)


@router.get("/sessions", response_model=list[SessionSummaryResponse])
async def list_sessions(
    access: RequestAccess = Depends(get_request_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> list[SessionSummaryResponse]:
    """列出当前 Workspace 的会话。"""
    summaries = service.list_sessions(
        target_workspace=access.target_workspace, access=access.access
    )
    return [SessionSummaryResponse.from_domain(summary) for summary in summaries]


@router.get("/sessions/{session_id}", response_model=ConversationSessionResponse)
async def get_session(
    session_id: str,
    access: RequestAccess = Depends(get_request_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSessionResponse:
    """读取会话全部轮次。"""
    session = service.get_session(
        session_id, target_workspace=access.target_workspace, access=access.access
    )
    return ConversationSessionResponse.from_session(session)


@router.patch("/sessions/{session_id}", response_model=ConversationSessionResponse)
async def rename_session(
    session_id: str,
    body: SessionTitleRequest,
    access: RequestAccess = Depends(get_request_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> ConversationSessionResponse:
    """重命名会话。"""
    session = service.rename_session(
        session_id, title=body.title, target_workspace=access.target_workspace, access=access.access
    )
    return ConversationSessionResponse.from_session(session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    access: RequestAccess = Depends(get_request_access),
    service: RootSessionService = Depends(get_root_session_service),
) -> Response:
    """删除空闲会话。"""
    service.delete_session(
        session_id, target_workspace=access.target_workspace, access=access.access
    )
    return Response(status_code=204)
