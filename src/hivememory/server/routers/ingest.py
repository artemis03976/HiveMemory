"""Passive ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.server.deps import get_ingress_service
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
    service: PassiveIngressService = Depends(get_ingress_service),
):
    """被动消息事件接入 HTTP 入口。"""
    event_fields = request.model_dump(
        exclude={"user_id", "agent_id"},
        exclude_none=True,
    )
    event = PassiveIngressEvent(**event_fields)
    result = await service.ingest_event(
        event=event,
        user_id=request.user_id,
        agent_id=request.agent_id,
    )
    return PassiveIngressResponse(**result)


@router.post("/ingest/flush", response_model=PassiveFlushResponse)
async def flush_conversation(
    request: PassiveFlushRequest,
    service: PassiveIngressService = Depends(get_ingress_service),
):
    """显式 seal 并提交指定外部会话的当前 turn。"""
    submitted = await service.flush_conversation(
        source=request.source,
        external_conversation_id=request.external_conversation_id,
        user_id=request.user_id,
        agent_id=request.agent_id,
    )
    return PassiveFlushResponse(submitted=submitted)
