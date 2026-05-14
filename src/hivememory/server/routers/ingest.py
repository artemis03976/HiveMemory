"""Passive ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.system import HiveMemorySystem
from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent
from hivememory.server.deps import get_system
from hivememory.server.models.ingest import (
    PassiveIngressRequest,
    PassiveIngressResponse,
)

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=PassiveIngressResponse)
async def ingest_event(
    request: PassiveIngressRequest,
    system: HiveMemorySystem = Depends(get_system),
):
    """被动消息事件接入 HTTP 入口，转调 HiveMemorySystem.ingest_event()。"""
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
    result = await system.ingest_event(
        event=event,
        user_id=request.user_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
    )
    return PassiveIngressResponse(**result)
