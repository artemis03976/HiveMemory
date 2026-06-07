"""Runtime data models for Alice agent execution.

PendingAtom / PendingAtomStatus / RuntimeScope 已上移到 ``core/models/pending.py``
（见 docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md §6.2），新代码请从 ``hivememory.core.models``
导入。本模块保留 alice runtime 自己的执行壳：
``MTPExecutionContext`` / ``ExecutionFrame`` / ``GenerationResult`` / ``StreamChunk``，
以及引擎↔编排解耦所需的执行信号 ``FrameExecutionResult`` / ``ExecutionProgress``
（见 docs/mod/AgentLoopDecouplingDesign.md §3.1 / §3.1bis）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity, RuntimeScope, TurnEvent
from hivememory.core.mtp.models import MTPCallRequest


@dataclass
class ExecutionProgress:
    """单帧执行的累积产物载体（PCB 的"程序状态"部分）。

    见 docs/mod/AgentLoopDecouplingDesign.md §3.1bis。
    write_foci / update_foci / pending_aliases 三个累积器已随 PendingAtomMaterializeTask
    重组移除（见 docs/mod/PendingAtomMaterializeTaskDesign.md §3.5）。
    """

    text_segments: List[str] = field(default_factory=list)
    turn_events: List[TurnEvent] = field(default_factory=list)
    iteration: int = 0
    sequence: int = 0


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

    # PCB 的"程序状态"：单帧执行的累积产物。Phase 1 起，引擎累积器从
    # execute_frame 的局部变量下沉到此处，使 CALL 挂起后重入续接、编号连续。
    # 见 docs/mod/AgentLoopDecouplingDesign.md §3.1bis。
    progress: "ExecutionProgress" = field(default_factory=ExecutionProgress)

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
    language: Optional[str] = None  # 显式语言覆盖；None 时由 runtime 从 agent_profile 派生


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


class FrameExecutionStatus(str, Enum):
    """引擎单次执行的停机原因。"""

    COMPLETED = "completed"   # 自然收敛
    SUSPENDED = "suspended"   # 命中 CALL，等待编排派生子 agent


@dataclass
class FrameExecutionResult:
    """引擎单次执行的 trap/return 信号。

    见 docs/mod/AgentLoopDecouplingDesign.md §3.1。它**不承载本帧累积产物**
    ——那些已下沉到 ``frame.progress``（见 ``ExecutionProgress``）。这里只表达
    "为什么停下来"，以及挂起时编排派生子帧所需的最小信息。

    引擎语义：``execute_frame(frame)`` 读写传入的 ``frame``，跑到自然收敛返回
    ``COMPLETED``，命中 CALL 返回 ``SUSPENDED`` 并把控制权交还编排，自己不 fork、
    不 resume、不组 CALL response。``AgentRunResult`` 不再由引擎产出，改由编排在 ``COMPLETED``
    时从 ``frame.progress`` 聚合。
    """

    status: FrameExecutionStatus

    # ---- status == SUSPENDED 时填充 ----
    # 触发 CALL 的派生请求（target_alias / task / context_refs）。
    call_request: Optional[MTPCallRequest] = None
    # WorkerAgent already normalizes the suspended MTP text with a right delimiter.
    suspend_assistant_text: Optional[str] = None
    # 供编排回填 tool_result TurnEvent 的 action_id。
    suspend_action_id: Optional[str] = None


__all__ = [
    "ExecutionFrame",
    "ExecutionProgress",
    "FrameExecutionResult",
    "FrameExecutionStatus",
    "GenerationResult",
    "MTPExecutionContext",
    "StreamChunk",
]
