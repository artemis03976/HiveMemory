"""Public Patchouli service contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity
from hivememory.core.protocol.models import AgentRunContext, EyeGazeResult


@dataclass
class StreamPrelude:
    """Data emitted before streaming Agent tokens."""

    topic_id: str
    is_new_topic: bool
    pool_snapshot: Dict[str, Any]
    memory_refs: List[Any]


@dataclass
class PreparedAgentRun:
    """Complete context prepared by Patchouli for one Agent run."""

    agent_run_context: AgentRunContext
    gaze_result: EyeGazeResult
    stream_prelude: StreamPrelude
    generation_options: Optional[Dict[str, Any]] = field(default=None)

    @property
    def identity(self) -> Identity:
        return self.agent_run_context.identity

    @property
    def agent_id(self) -> str:
        return self.agent_run_context.identity.agent_id

    @property
    def topic_id(self) -> str:
        return self.agent_run_context.topic_id

    @property
    def user_message(self) -> str:
        return self.agent_run_context.user_message

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_run_context.agent_profile
