"""Agent 请求/响应模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from hivememory.core.models import MemoryAtom


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
