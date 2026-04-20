"""Agents 路由 — Agent Profile 列表"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from hivememory.patchouli.system import PatchouliSystem
from hivememory.server.deps import get_system

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


@router.post("/agents", response_model=AgentProfileResponse, status_code=201)
async def create_agent(body: AgentCreateRequest, system: PatchouliSystem = Depends(get_system)):
    """创建新的 Agent Profile"""
    from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, Artifacts, MemoryType
    atom = MemoryAtom(
        meta=MetaData(source_agent_id="ui", user_id="default"),
        index=IndexLayer(
            title=body.title,
            summary=body.summary or body.title,
            tags=body.tags,
            memory_type=MemoryType.AGENT_PROFILE,
            alias=body.alias,
        ),
        payload=PayloadLayer(
            content=body.content,
            artifacts=Artifacts(agent_config=body.agent_config),
        ),
    )
    try:
        system.storage.upsert_memory(atom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AgentProfileResponse(
        id=str(atom.id),
        alias=atom.index.alias or str(atom.id),
        title=atom.index.title,
        summary=atom.index.summary,
        tags=atom.index.tags,
        content=atom.payload.content,
        agent_config=atom.payload.artifacts.agent_config,
    )


@router.get("/agents", response_model=List[AgentProfileResponse])
async def list_agents(system: PatchouliSystem = Depends(get_system)):
    """列出所有 Agent Profile"""
    try:
        atoms = system.storage.get_all_memories(
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=100,
        )
        return [
            AgentProfileResponse(
                id=str(atom.id),
                alias=atom.index.alias or str(atom.id),
                title=atom.index.title,
                summary=atom.index.summary,
                tags=atom.index.tags,
                content=atom.payload.content,
                agent_config=atom.payload.artifacts.agent_config,
            )
            for atom in atoms
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
