"""Agents 路由 — Agent Profile 列表"""

from fastapi import APIRouter, Depends

from hivememory.server.deps import RequestAccess, get_request_access, get_root_agent_service
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.agent import AgentCreateRequest, AgentProfileResponse
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["agents"])


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    service: RootAgentProfileService = Depends(get_root_agent_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """创建新的 Agent Profile（管理用例，actor 为保留 system）"""
    atom = await service.create_agent_profile(
        target_workspace=request_access.target_workspace,
        title=body.title,
        alias=body.alias,
        summary=body.summary,
        content=body.content,
        tags=body.tags,
        agent_config=body.agent_config,
        access=request_access.access,
    )
    return AgentProfileResponse.from_atom(atom)


@router.get("/agents", response_model=list[AgentProfileResponse])
async def list_agents(
    service: RootAgentProfileService = Depends(get_root_agent_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """列出所有 Agent Profile（管理用例，actor 为保留 system）"""
    atoms = await service.list_agent_profiles(
        target_workspace=request_access.target_workspace,
        limit=100,
        access=request_access.access,
    )
    return [AgentProfileResponse.from_atom(atom) for atom in atoms]
