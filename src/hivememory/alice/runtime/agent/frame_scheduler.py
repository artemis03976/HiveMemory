"""Backward-compatible frame construction facade.

Frame ownership and suspension state now live in ``RunSession``.  This class
remains temporarily so older integrations can migrate without keeping a
process-wide frame stack.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import uuid4

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.alice.runtime.agent.frame_factory import FrameFactory, FrameSpec
from hivememory.core.models import AgentProfile, Identity, RuntimeScope

if TYPE_CHECKING:
    from hivememory.prompts.assembler import AgentPromptAssembler


class FrameScheduler:
    """Compatibility shell delegating construction to a stateless factory."""

    def __init__(self, prompt_assembler: AgentPromptAssembler) -> None:
        self._prompt_assembler = prompt_assembler
        self._factory = FrameFactory()
        self._compat_suspended: ContextVar[ExecutionFrame | None] = ContextVar(
            "alice_compat_suspended_frame", default=None
        )

    def create_main_frame(
        self,
        agent_profile: AgentProfile,
        messages: list[dict],
        topic_id: str,
        identity: Identity,
        *,
        run_id: str | None = None,
        execution_policy: FrameExecutionPolicy | None = None,
    ) -> ExecutionFrame:
        run_id = run_id or f"run_{uuid4().hex}"
        return self._factory.create(
            FrameSpec(
                runtime_scope=self._factory.scope(
                    run_id=run_id,
                    frame_id=f"frame_main_{uuid4().hex}",
                ),
                profile=agent_profile,
                identity=identity,
                messages=messages,
                topic_id=topic_id,
                execution_policy=execution_policy or FrameExecutionPolicy(),
            )
        )

    async def fork_sub_frame(
        self,
        parent_frame: ExecutionFrame,
        agent_profile: AgentProfile,
        task: str,
        shared_context: str = "",
        *,
        execution_policy: FrameExecutionPolicy | None = None,
    ) -> ExecutionFrame:
        messages = self._prompt_assembler.build_sub_agent_messages(
            profile=agent_profile,
            task=task,
            shared_context=shared_context,
            depth=1,
        )
        # parent/depth are retained only as compatibility metadata until
        # Phase 6 removes the legacy RuntimeScope fields.
        scope = RuntimeScope(
            run_id=parent_frame.runtime_scope.run_id,
            frame_id=f"frame_sub_{uuid4().hex}",
            parent_frame_id=parent_frame.runtime_scope.frame_id,
            depth=parent_frame.runtime_scope.depth + 1,
        )
        return self._factory.create(
            FrameSpec(
                runtime_scope=scope,
                profile=agent_profile,
                identity=parent_frame.identity,
                messages=messages,
                topic_id=None,
                execution_policy=execution_policy or FrameExecutionPolicy(),
            )
        )

    def suspend_frame(self, frame: ExecutionFrame) -> None:
        """Compatibility hook; suspension is no longer scheduler-owned."""
        self._compat_suspended.set(frame)

    def resume_frame(self) -> ExecutionFrame | None:
        """Return the compatibility frame without consulting a shared stack."""
        frame = self._compat_suspended.get()
        self._compat_suspended.set(None)
        return frame

    def get_current_depth(self) -> int:
        return 1 if self._compat_suspended.get() is not None else 0

    def clear_stack(self) -> None:
        self._compat_suspended.set(None)


__all__ = ["FrameScheduler"]
