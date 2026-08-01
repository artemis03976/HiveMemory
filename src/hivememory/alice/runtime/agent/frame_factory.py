from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import AgentProfile, Identity, RuntimeScope


@dataclass(frozen=True)
class FrameSpec:
    runtime_scope: RuntimeScope
    profile: AgentProfile
    identity: Identity
    messages: Sequence[dict[str, str]]
    topic_id: str | None
    execution_policy: FrameExecutionPolicy


class FrameFactory:
    """Stateless constructor for ordinary execution frames."""

    def create(self, spec: FrameSpec) -> ExecutionFrame:
        return ExecutionFrame(
            runtime_scope=spec.runtime_scope,
            agent_profile=spec.profile,
            working_history=[dict(message) for message in spec.messages],
            topic_id=spec.topic_id,
            identity=spec.identity,
            execution_policy=spec.execution_policy,
        )

    @staticmethod
    def scope(*, run_id: str, frame_id: str | None = None) -> RuntimeScope:
        return RuntimeScope(run_id=run_id, frame_id=frame_id or f"frame_{uuid4().hex}")


__all__ = ["FrameFactory", "FrameSpec"]
