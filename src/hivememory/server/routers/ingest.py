"""Passive ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.core.models import IdentityScope
from hivememory.server.deps import (
    RequestIdentitySelection,
    get_identity_selection,
    get_ingress_service,
    resolve_request_identity_scope,
)
from hivememory.server.models.ingest import (
    PassiveFlushRequest,
    PassiveFlushResponse,
    PassiveIngressRequest,
    PassiveIngressResponse,
)
from hivememory.system.services.passive import PassiveIngressEvent
from hivememory.system.application.passive_ingress_service import PassiveIngressService

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=PassiveIngressResponse)
async def ingest_event(
    request: PassiveIngressRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: PassiveIngressService = Depends(get_ingress_service),
):
    """被动消息事件接入 HTTP 入口。

    被动接入是 Agent-bearing action：connector 必须显式携带具体
    ``agent_id``（无默认值），它参与外部会话命名空间；body 身份与统一
    请求头冲突时显式拒绝。
    """
    identity_scope = resolve_request_identity_scope(
        selection,
        require_agent=True,
        agent_id=request.agent_id,
        explicit_user_id=request.user_id,
        explicit_workspace_id=request.workspace_id,
    )
    event_fields = request.model_dump(
        exclude={"user_id", "workspace_id", "agent_id"},
        exclude_none=True,
    )
    event = PassiveIngressEvent(**event_fields)
    result = await service.ingest_event(event=event, identity_scope=identity_scope)
    return PassiveIngressResponse(**result)


@router.post("/ingest/flush", response_model=PassiveFlushResponse)
async def flush_conversation(
    request: PassiveFlushRequest,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: PassiveIngressService = Depends(get_ingress_service),
):
    """显式 seal 并提交指定外部会话的当前 turn。"""
    identity_scope = resolve_request_identity_scope(
        selection,
        require_agent=True,
        agent_id=request.agent_id,
        explicit_user_id=request.user_id,
        explicit_workspace_id=request.workspace_id,
    )
    submitted = await service.flush_conversation(
        source=request.source,
        external_conversation_id=request.external_conversation_id,
        identity_scope=identity_scope,
    )
    return PassiveFlushResponse(submitted=submitted)
