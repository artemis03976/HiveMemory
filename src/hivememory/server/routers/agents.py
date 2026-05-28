"""Agents 路由 — Agent Profile 列表"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hivememory.core.models import MemoryAtom
from hivememory.server.deps import get_agent_service
from hivememory.system.application.agent_service import AgentApplicationService

router = APIRouter(tags=["agents"])


class AgentCreateRequest(BaseModel):
    title: str
    alias: str
    summary: str = ""
    content: str = ""
    tags: List[str] = []
    agent_config: Optional[Dict[str, Any]] = None


class AgentProfileResponse(BaseModel):
    id: str
    alias: str
    title: str
    summary: str
    tags: List[str]
    content: str = ""  # payload.content — Agent 的人格/系统指令
    agent_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_atom(cls, atom: MemoryAtom) -> "AgentProfileResponse":
        return cls(
            id=str(atom.id),
            alias=atom.index.alias or str(atom.id),
            title=atom.index.title,
            summary=atom.index.summary,
            tags=atom.index.tags,
            content=atom.payload.content,
            agent_config=atom.payload.artifacts.agent_config,
        )


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    service: AgentApplicationService = Depends(get_agent_service),
):
    """创建新的 Agent Profile"""
    try:
        atom = service.create_agent_profile(
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
        atoms = service.list_agent_profiles(limit=100)
        return [AgentProfileResponse.from_atom(atom) for atom in atoms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
