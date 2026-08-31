"""Alice 单 Agent 执行层的运行时数据模型。

PendingAtom / PendingAtomStatus / RuntimeScope 已上移到 ``core/models/pending.py``
（见 docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md §6.2），新代码请从 ``hivememory.core.models``
导入。本模块保留 agent_runtime 自己的执行壳：
``MTPExecutionContext`` / ``ExecutionFrame`` / ``GenerationResult`` / ``StreamChunk``，
以及引擎↔编排解耦所需的执行信号 ``FrameExecutionResult`` / ``ExecutionProgress``
（见 docs/archive/plans/implementation/agent-loop-decoupling.md §3.1 / §3.1bis）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import (
    AgentProfile,
    Identity,
    RuntimeScope,
    TurnEvent,
    IdentityScope,
)
from hivememory.core.mtp.models import MTPCallRequest


@dataclass
class ExecutionProgress:
    """单帧执行的累积产物载体（PCB 的"程序状态"部分）。

    见 docs/archive/plans/implementation/agent-loop-decoupling.md §3.1bis。
    write_foci / update_foci / pending_aliases 三个累积器已随 PendingAtomMaterializeTask
    重组移除（见 docs/archive/legacy-docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md §3.5）。
    """

    text_segments: list[str] = field(default_factory=list)
    turn_events: list[TurnEvent] = field(default_factory=list)
    iteration: int = 0
    sequence: int = 0
    # 本帧实际使用的模型展示名（由 AgentRuntime 在 run_frame 开始时写入，
    # 来自 ModelRegistry.resolve()；前端话题信息展示该值）
    model_used: str = ""


@dataclass
class ExecutionFrame:
    """
    一次单 Agent 执行的可恢复进程控制块（PCB）。

    ExecutionFrame 携带运行一个 Agent 所需的隔离状态，使共享 runtime 无需
    保存 per-frame 的 identity 与权限。执行进度（正文、事件、迭代与序号）
    全部放在 ``progress`` 上，CALL 挂起后重入同一 frame 可自然续接
    （见 docs/alice/agent-runtime.md §2）。
    """

    runtime_scope: RuntimeScope
    agent_profile: AgentProfile
    working_history: list[dict[str, str]]
    topic_id: str | None
    execution_policy: FrameExecutionPolicy = field(default_factory=FrameExecutionPolicy)

    harvested_aliases: list[str] = field(default_factory=list)

    # PCB 的"程序状态"：单帧执行的累积产物。Phase 1 起，引擎累积器从
    # execute_frame 的局部变量下沉到此处，使 CALL 挂起后重入续接、编号连续。
    # 见 docs/archive/plans/implementation/agent-loop-decoupling.md §3.1bis。
    progress: ExecutionProgress = field(default_factory=ExecutionProgress)

    @property
    def identity_scope(self) -> IdentityScope:
        """返回 frame 随 RuntimeScope 继承的 Workspace hard boundary。"""
        return self.runtime_scope.identity_scope

    @property
    def identity(self) -> Identity:
        """兼容读取当前执行者身份。"""
        return self.identity_scope.actor_identity

    def is_transient(self) -> bool:
        """判断本帧是否未挂载话题（子帧为瞬态帧，topic_id 为 None）。"""
        return self.topic_id is None

    def add_harvested_alias(self, alias: str) -> None:
        """去重登记一次 WRITE/UPDATE 产生的 alias（供 finalize_frame 收割）。"""
        if alias and alias not in self.harvested_aliases:
            self.harvested_aliases.append(alias)

    def __repr__(self) -> str:
        return (
            f"ExecutionFrame(frame={self.runtime_scope.frame_id}, "
            f"agent={self.agent_profile.model_name}, "
            f"topic={self.topic_id}, "
            f"harvested={len(self.harvested_aliases)})"
        )


@dataclass(frozen=True)
class MTPExecutionContext:
    """单条 MTP 指令执行时的身份、权限与运行坐标上下文。"""

    runtime_scope: RuntimeScope
    agent_profile: Any = None
    execution_policy: FrameExecutionPolicy | None = None
    language: str | None = None  # 显式语言覆盖；None 时由 runtime 从 agent_profile 派生

    @property
    def identity_scope(self) -> IdentityScope:
        """返回 MTP 指令继承的 Workspace hard boundary。"""
        return self.runtime_scope.identity_scope

    @property
    def identity(self) -> Identity:
        """兼容读取当前执行者身份。"""
        return self.identity_scope.actor_identity


@dataclass
class GenerationResult:
    """一次 LLM 生成的结构化结果。"""

    text: str = ""
    finish_reason: str = "stop"
    was_mtp_interrupted: bool = False
    prefix_text: str = ""
    mtp_fragment: str = ""
    # 实际发给 litellm 的模型标识符（如 "deepseek/deepseek-chat"）
    # 由 WorkerAgentService._extract_runtime_params 解析后写入
    model_used: str = ""


@dataclass
class StreamChunk:
    """流式生成发出的单个 chunk。"""

    delta: str = ""
    full_text: str = ""
    is_final: bool = False
    result: GenerationResult | None = None
    mtp_detected: bool = False


class FrameExecutionStatus(str, Enum):
    """引擎单次执行的停机原因。"""

    COMPLETED = "completed"  # 自然收敛
    SUSPENDED = "suspended"  # 命中 CALL，等待编排派生子 agent
    CANCELLED = "cancelled"  # 收到取消信号，提前退出
    FAILED = "failed"  # 本帧无法形成有效终态
    BUDGET_EXHAUSTED = "budget_exhausted"  # 达到循环预算但没有自然收敛


@dataclass
class FrameExecutionResult:
    """引擎单次执行的 trap/return 信号。

    见 docs/archive/plans/implementation/agent-loop-decoupling.md §3.1。它**不承载本帧累积产物**
    ——那些已下沉到 ``frame.progress``（见 ``ExecutionProgress``）。这里只表达
    "为什么停下来"，以及挂起时编排派生子帧所需的最小信息。

    引擎语义：``execute_frame(frame)`` 读写传入的 ``frame``，跑到自然收敛返回
    ``COMPLETED``，命中 CALL 返回 ``SUSPENDED`` 并把控制权交还编排，取消、失败和
    循环预算耗尽分别返回对应终态。引擎自己不 fork、不 resume、不组 CALL response。
    ``AgentRunResult`` 不再由引擎产出，改由编排在 frame 终态确定后从 ``frame.progress``
    聚合。
    """

    status: FrameExecutionStatus

    # ---- status == SUSPENDED 时填充 ----
    # 触发 CALL 的派生请求（target_alias / task / context_refs）。
    call_request: MTPCallRequest | None = None
    # WorkerAgent 已为被 stop sequence 截断的挂起文本补全右定界符。
    suspend_assistant_text: str | None = None
    # 供编排回填 tool_result TurnEvent 的 action_id。
    suspend_action_id: str | None = None
    # FAILED 时保留受控的内部异常，编排层只把稳定错误信息回填给 Agent。
    error: Exception | None = None


__all__ = [
    "ExecutionFrame",
    "ExecutionProgress",
    "FrameExecutionResult",
    "FrameExecutionStatus",
    "GenerationResult",
    "MTPExecutionContext",
    "StreamChunk",
]
