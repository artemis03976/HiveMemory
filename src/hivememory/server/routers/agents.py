"""Agents 路由 — Agent Profile 列表"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.server.deps import get_agent_service
from hivememory.server.models.agent import AgentCreateRequest, AgentProfileResponse
from hivememory.system.application.agent_service import AgentApplicationService

router = APIRouter(tags=["agents"])


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    service: AgentApplicationService = Depends(get_agent_service),
):
    """创建新的 Agent Profile"""
    try:
        atom = await service.create_agent_profile(
            title=body.title,
            alias=body.alias,
            summary=body.summary,
            content=body.content,
            tags=body.tags,
            agent_config=body.agent_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AgentProfileResponse.from_atom(atom)


@router.get("/agents", response_model=List[AgentProfileResponse])
async def list_agents(
    service: AgentApplicationService = Depends(get_agent_service),
):
    """列出所有 Agent Profile"""
    try:
        atoms = await service.list_agent_profiles(limit=100)
        return [AgentProfileResponse.from_atom(atom) for atom in atoms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
