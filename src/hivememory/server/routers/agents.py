"""Agents 路由 — Agent Profile 列表"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.core.errors import InvalidMemoryFieldError, MemoryAliasConflictError
from hivememory.server.deps import RequestAccess, get_agent_service, get_request_access
from hivememory.server.models.agent import AgentCreateRequest, AgentProfileResponse
from hivememory.workspace.capability.agent_profiles import AgentApplicationService

router = APIRouter(tags=["agents"])


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    service: AgentApplicationService = Depends(get_agent_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """创建新的 Agent Profile（管理用例，actor 为保留 system）"""
    try:
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
    except InvalidMemoryFieldError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MemoryAliasConflictError as exc:
        # Agent alias 即 agent_id，同一 Workspace 内必须唯一。
        raise HTTPException(status_code=409, detail=str(exc))
    return AgentProfileResponse.from_atom(atom)


@router.get("/agents", response_model=list[AgentProfileResponse])
async def list_agents(
    service: AgentApplicationService = Depends(get_agent_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """列出所有 Agent Profile（管理用例，actor 为保留 system）"""
    atoms = await service.list_agent_profiles(
        target_workspace=request_access.target_workspace,
        limit=100,
        access=request_access.access,
    )
    return [AgentProfileResponse.from_atom(atom) for atom in atoms]
