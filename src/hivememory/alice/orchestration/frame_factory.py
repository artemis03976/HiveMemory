from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from hivememory.agent_runtime.models import ExecutionFrame, ExecutionProgress
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import (
    AgentProfile,
    RuntimeScope,
    TurnEvent,
    IdentityScope,
)


@dataclass(frozen=True)
class FrameSpec:
    """创建 ExecutionFrame 的不可变规格。"""

    runtime_scope: RuntimeScope
    profile: AgentProfile
    messages: Sequence[dict[str, str]]
    topic_id: str | None
    execution_policy: FrameExecutionPolicy


class FrameFactory:
    """无状态创建普通 ExecutionFrame 的工厂。

    不表达主/子拓扑：父子关系只记录在 RunSession 的 frame registry 与
    CallRecord 中（见 docs/alice/orchestration.md §2）。
    """

    def create(self, spec: FrameSpec) -> ExecutionFrame:
        return ExecutionFrame(
            runtime_scope=spec.runtime_scope,
            agent_profile=spec.profile,
            working_history=[dict(message) for message in spec.messages],
            topic_id=spec.topic_id,
            execution_policy=spec.execution_policy,
            progress=self._initial_progress(spec.messages),
        )

    @staticmethod
    def _initial_progress(messages: Sequence[dict[str, str]]) -> ExecutionProgress:
        """把最新一条 user 消息作为 TurnEvent 写入新帧日志首位。"""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            if not content:
                return ExecutionProgress()
            return ExecutionProgress(
                turn_events=[
                    TurnEvent(
                        kind="user_message",
                        sequence=0,
                        role="user",
                        content=content,
                    )
                ],
                sequence=1,
            )
        return ExecutionProgress()

    @staticmethod
    def scope(
        *,
        identity_scope: IdentityScope,
        run_id: str,
        frame_id: str | None = None,
    ) -> RuntimeScope:
        """生成继承 hard boundary 的唯一 run/frame 坐标。"""
        return RuntimeScope(
            identity_scope=identity_scope,
            run_id=run_id,
            frame_id=frame_id or f"frame_{uuid4().hex}",
        )


__all__ = ["FrameFactory", "FrameSpec"]
