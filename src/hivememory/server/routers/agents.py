"""Agents 路由 — Agent Profile 的新建、列出、编辑与删除（管理用例，actor 为保留 system）"""

from fastapi import APIRouter, Depends

from hivememory.core.access import WorkingActor
from hivememory.core.errors import ResourceNotFoundError
from hivememory.server.deps import get_request_actor, get_root_agent_service
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.agent import (
    AgentCreateRequest,
    AgentProfileResponse,
    AgentUpdateRequest,
)
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["agents"])


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    service: RootAgentProfileService = Depends(get_root_agent_service),
    actor: WorkingActor = Depends(get_request_actor),
):
    """新建 Agent Profile；同名已存在时返回 409，不覆盖"""
    stored = await service.create_agent_profile(
        body.agent_id,
        body.fields(),
        target_workspace=actor.workspace,
        actor=actor,
    )
    return AgentProfileResponse.from_stored(stored)


@router.get("/agents", response_model=list[AgentProfileResponse])
async def list_agents(
    service: RootAgentProfileService = Depends(get_root_agent_service),
    actor: WorkingActor = Depends(get_request_actor),
):
    """按 agent_id 列出全部合格的 Agent Profile，含停用的"""
    profiles = await service.list_agent_profiles(target_workspace=actor.workspace, actor=actor)
    return [AgentProfileResponse.from_stored(stored) for stored in profiles]


@router.patch("/agents/{agent_id}", response_model=AgentProfileResponse)
async def update_agent(
    agent_id: str,
    body: AgentUpdateRequest,
    service: RootAgentProfileService = Depends(get_root_agent_service),
    actor: WorkingActor = Depends(get_request_actor),
):
    """修改 Agent Profile 中请求给出的字段；agent_id 不可改"""
    stored = await service.update_agent_profile(
        agent_id,
        body.changes(),
        target_workspace=actor.workspace,
        actor=actor,
    )
    return AgentProfileResponse.from_stored(stored)


@router.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    service: RootAgentProfileService = Depends(get_root_agent_service),
    actor: WorkingActor = Depends(get_request_actor),
):
    """删除 Agent Profile"""
    deleted = await service.delete_agent_profile(
        agent_id, target_workspace=actor.workspace, actor=actor
    )
    if not deleted:
        raise ResourceNotFoundError()
