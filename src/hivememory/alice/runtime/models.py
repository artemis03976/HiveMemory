"""Runtime data models for Alice agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from hivememory.alice.runtime.pending_atom_state import PendingAtomStatus
from hivememory.core.models import AgentProfile, Identity
from hivememory.engines.generation.models import (
    PendingAtomSettlement,
    UpdateFocus,
    WriteFocus,
)
from pydantic import BaseModel, ConfigDict, Field


class RuntimeScope(BaseModel):
    """Runtime execution coordinates for an Alice agent run."""

    run_id: str = ""
    frame_id: str = ""
    parent_frame_id: Optional[str] = None
    action_id: Optional[str] = None
    depth: int = 0

    def with_action(self, action_id: str) -> "RuntimeScope":
        """Return a copy scoped to one agent action."""
        return self.model_copy(update={"action_id": action_id})

    def for_child(self, frame_id: str) -> "RuntimeScope":
        """Return a child frame scope under the same run."""
        return RuntimeScope(
            run_id=self.run_id,
            frame_id=frame_id,
            parent_frame_id=self.frame_id,
            depth=self.depth + 1,
        )

    model_config = ConfigDict(frozen=True)


class PendingAtom(BaseModel):
    """
    运行时待物化记忆句柄。

    不是正式 MemoryAtom，不承诺最终落库。
    在其生命周期内，Agent 可通过 pending_alias 读取本次写入意图的内容。
    """

    pending_alias: str
    intent_id: Optional[str] = None
    status: PendingAtomStatus
    source_verb: Literal["WRITE", "UPDATE"]

    focus: WriteFocus | UpdateFocus
    identity: Identity = Field(default_factory=Identity)
    runtime_scope: RuntimeScope = Field(default_factory=RuntimeScope)
    created_at: datetime = Field(default_factory=datetime.now)

    # Phase 2: settlement tracking
    settlement: Optional[PendingAtomSettlement] = None


@dataclass
class ExecutionFrame:
    """
    Runtime frame for one agent generation loop.

    An ExecutionFrame carries the isolated state needed to run a main agent or
    sub-agent frame without storing per-frame identity and permissions on the
    shared runtime.
    """

    runtime_scope: RuntimeScope
    agent_profile: AgentProfile
    working_history: List[Dict[str, str]]
    topic_id: Optional[str]
    identity: Identity

    harvested_aliases: List[str] = field(default_factory=list)

    def is_main_frame(self) -> bool:
        """Return True when this frame belongs to the main agent."""
        return self.runtime_scope.depth == 0

    def is_sub_frame(self) -> bool:
        """Return True when this frame belongs to a sub-agent."""
        return self.runtime_scope.depth >= 1

    def is_transient(self) -> bool:
        """Return True when this frame is not mounted to a topic."""
        return self.topic_id is None

    def add_harvested_alias(self, alias: str) -> None:
        """Record a WRITE/UPDATE alias once."""
        if alias and alias not in self.harvested_aliases:
            self.harvested_aliases.append(alias)

    def __repr__(self) -> str:
        return (
            f"ExecutionFrame(frame={self.runtime_scope.frame_id}, "
            f"agent={self.agent_profile.model_name}, "
            f"depth={self.runtime_scope.depth}, "
            f"topic={self.topic_id}, "
            f"harvested={len(self.harvested_aliases)})"
        )


@dataclass(frozen=True)
class MTPExecutionContext:
    """Identity and permission context for a single MTP command execution."""

    identity: Identity = field(default_factory=Identity)
    agent_profile: Any = None
    runtime_scope: RuntimeScope = field(default_factory=RuntimeScope)


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
    "PendingAtom",
    "PendingAtomStatus",
    "RuntimeScope",
    "StreamChunk",
]
