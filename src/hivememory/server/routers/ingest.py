"""Passive ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.system.application.passive import PassiveIngressEvent
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.server.deps import get_ingress_service
from hivememory.server.models.ingest import (
    PassiveIngressRequest,
    PassiveIngressResponse,
)

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=PassiveIngressResponse)
async def ingest_event(
    request: PassiveIngressRequest,
    service: PassiveIngressService = Depends(get_ingress_service),
):
    """被动消息事件接入 HTTP 入口。"""
    event = PassiveIngressEvent(
        role=request.role,
        content=request.content,
        action_id=request.action_id,
        tool_name=request.tool_name,
        tool_kind=request.tool_kind,
        tool_args=request.tool_args,
        target=request.target,
        status=request.status,
        render_as=request.render_as,
    )
    result = await service.ingest_event(
        event=event,
        user_id=request.user_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
    )
    return PassiveIngressResponse(**result)
