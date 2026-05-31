"""Runtime data models for Alice agent execution.

PendingAtom / RuntimeScope 已上移到 ``core/models/pending.py``（见
docs/mod/PendingAtomRuntimeDesign.md §6.2）。本模块保留 alice runtime 自己的
执行壳（``MTPExecutionContext`` / ``ExecutionFrame`` / ``GenerationResult`` /
``StreamChunk``），并对已迁出的领域模型做 re-export 兼容，避免一次性触动所有引用方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity
from hivememory.core.models.pending import (
    PendingAtom,
    PendingAtomStatus,
    RuntimeScope,
)


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
