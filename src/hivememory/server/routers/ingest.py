"""Ingest 路由 — POST /api/v1/ingest"""

from fastapi import APIRouter, Depends

from hivememory.patchouli.system import PatchouliSystem
from hivememory.server.deps import get_system
from hivememory.server.models.ingest import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    system: PatchouliSystem = Depends(get_system),
):
    """被动消息摄入 — 封装 PatchouliSystem.ingest()"""
    result = await system.ingest(
        role=request.role,
        content=request.content,
        user_id=request.user_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
        context=request.context,
    )
    return IngestResponse(**result)
