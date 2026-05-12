"""Ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent
from hivememory.server.deps import get_system
from hivememory.server.models.ingest import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    system: PatchouliSystem = Depends(get_system),
):
    """被动消息摄入 — 封装 PatchouliSystem.ingest_event()"""
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
    return IngestResponse(**result)
