"""Agents 路由 — Agent Profile 列表"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from hivememory.patchouli.system import PatchouliSystem
from hivememory.server.deps import get_system

router = APIRouter(tags=["agents"])


class AgentProfileResponse(BaseModel):
    id: str
    alias: str
    title: str
    summary: str
    tags: List[str]
    content: str = ""  # payload.content — Agent 的人格/系统指令
    agent_config: Optional[Dict[str, Any]] = None


@router.get("/agents", response_model=List[AgentProfileResponse])
async def list_agents(system: PatchouliSystem = Depends(get_system)):
    """列出所有 Agent Profile"""
    try:
        atoms = system.storage.get_all_memories(
            filters={"memory_type": "AGENT_PROFILE"},
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
