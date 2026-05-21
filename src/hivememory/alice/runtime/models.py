"""Runtime data models for Alice agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity


@dataclass
class ExecutionFrame:
    """
    Runtime frame for one agent generation loop.

    An ExecutionFrame carries the isolated state needed to run a main agent or
    sub-agent frame without storing per-frame identity and permissions on the
    shared runtime.
    """

    process_id: str
    agent_profile: AgentProfile
    working_history: List[Dict[str, str]]
    depth: int
    topic_id: Optional[str]
    identity: Identity

    parent_frame_id: Optional[str] = None
    harvested_aliases: List[str] = field(default_factory=list)

    def is_main_frame(self) -> bool:
        """Return True when this frame belongs to the main agent."""
        return self.depth == 0

    def is_sub_frame(self) -> bool:
        """Return True when this frame belongs to a sub-agent."""
        return self.depth >= 1

    def is_transient(self) -> bool:
        """Return True when this frame is not mounted to a topic."""
        return self.topic_id is None

    def add_harvested_alias(self, alias: str) -> None:
        """Record a WRITE/UPDATE alias once."""
        if alias and alias not in self.harvested_aliases:
            self.harvested_aliases.append(alias)

    def __repr__(self) -> str:
        return (
            f"ExecutionFrame(pid={self.process_id}, "
            f"agent={self.agent_profile.model_name}, "
            f"depth={self.depth}, "
            f"topic={self.topic_id}, "
            f"harvested={len(self.harvested_aliases)})"
        )


@dataclass(frozen=True)
class MTPExecutionContext:
    """Identity and permission context for a single MTP command execution."""

    identity: Identity = field(default_factory=Identity)
    agent_profile: Any = None
    depth: int = 0


@dataclass
class GenerationResult:
    """Structured result for one LLM generation."""

    text: str = ""
    finish_reason: str = "stop"
    was_mtp_interrupted: bool = False
    prefix_text: str = ""
    mtp_fragment: str = ""


@dataclass
class StreamChunk:
    """Single chunk emitted by streaming generation."""

    delta: str = ""
    full_text: str = ""
    is_final: bool = False
    result: Optional[GenerationResult] = None
    mtp_detected: bool = False


__all__ = [
    "ExecutionFrame",
    "GenerationResult",
    "MTPExecutionContext",
    "StreamChunk",
]
