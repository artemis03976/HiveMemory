"""Alice 单 Agent 执行层的运行时数据模型。

PendingAtom / PendingAtomStatus / RuntimeScope 由 ``hivememory.core.models`` 提供，
写入意图登记由 workspace 持有。本模块保留 agent_runtime 自己的执行壳：
``MTPExecutionContext`` / ``ExecutionFrame`` / ``GenerationResult`` / ``StreamChunk``，
以及引擎↔编排解耦所需的执行信号 ``FrameExecutionResult`` / ``ExecutionProgress``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import (
    AgentProfile,
    RuntimeScope,
    TurnEvent,
)
from hivememory.core.mtp.models import MTPCallRequest

if TYPE_CHECKING:
    from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
    from hivememory.workspace.contracts import OperationSubmitter


@dataclass
class ExecutionProgress:
    """单帧执行的累积产物载体（PCB 的"程序状态"部分）。

    累积器放在 frame 上而不是循环的局部变量中，CALL 挂起后重入同一 frame
    时自然续接，事件编号连续。写入意图的登记不在这里累积，由 workspace 负责。
    """

    text_segments: list[str] = field(default_factory=list)
    # 本次 LLM 调用已交付的文字；取消或失败时封成未完成消息。
    streaming_text: list[str] = field(default_factory=list)
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
    保存每帧的观测标签与权限。资源访问由进程绑定的操作提交函数负责，
    标签仅用于展示与观测。执行进度（正文、事件、迭代与序号）
    全部放在 ``progress`` 上，CALL 挂起后重入同一 frame 可自然续接。
    """

    runtime_scope: RuntimeScope
    agent_profile: AgentProfile
    working_history: list[dict[str, str]]
    topic_id: str | None
    execution_policy: FrameExecutionPolicy = field(default_factory=FrameExecutionPolicy)

    # 本进程的操作提交函数由 root 与 CALL 子帧共享，线程 context 由 CPU 驱动绑定。
    submit_operation: OperationSubmitter | None = None

    harvested_aliases: list[str] = field(default_factory=list)

    # PCB 的"程序状态"：单帧执行的累积产物。累积器放在此处而不是
    # execute_frame 的局部变量中，使 CALL 挂起后重入续接、编号连续。
    progress: ExecutionProgress = field(default_factory=ExecutionProgress)
    # 预算随实际模型解析一次；CALL 恢复时继续使用同一组模型参数与预算。
    resolved_model: ResolvedFrameModel | None = None

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
    """单条 MTP 指令执行时的观测标签、权限与运行坐标上下文。"""

    runtime_scope: RuntimeScope
    submit_operation: OperationSubmitter | None = None
    agent_profile: Any = None
    execution_policy: FrameExecutionPolicy | None = None
    language: str | None = None  # 显式语言覆盖；None 时由 runtime 从 agent_profile 派生


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


@dataclass(frozen=True)
class CallSuspension:
    """CALL 挂起中已确认存在的派生请求与 action_id。"""

    call_request: MTPCallRequest
    action_id: str


@dataclass
class FrameExecutionResult:
    """引擎单次执行的 trap/return 信号。

    它**不承载本帧累积产物**
    ——那些已下沉到 ``frame.progress``（见 ``ExecutionProgress``）。这里只表达
    "为什么停下来"，以及挂起时编排派生子帧所需的最小信息。

    引擎语义：``execute_frame(frame)`` 读写传入的 ``frame``，跑到自然收敛返回
    ``COMPLETED``，命中 CALL 返回 ``SUSPENDED`` 并把控制权交还编排，取消、失败和
    循环预算耗尽分别返回对应终态。引擎自己不 fork、不 resume、不组 CALL response。
    执行结果（CPU 执行结果）不再由引擎产出，改由编排在 frame 终态确定后从 ``frame.progress``
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

    def require_call_suspension(self) -> CallSuspension:
        """取出 CALL 挂起的派生信息；不是完整的 CALL 挂起时显式失败。"""
        if self.status != FrameExecutionStatus.SUSPENDED:
            raise RuntimeError("CALL suspension requires a suspended frame result.")
        if self.call_request is None:
            raise RuntimeError("CALL suspension is missing its request.")
        if not self.suspend_action_id:
            raise RuntimeError("CALL suspension is missing its action id.")
        return CallSuspension(call_request=self.call_request, action_id=self.suspend_action_id)


__all__ = [
    "CallSuspension",
    "ExecutionFrame",
    "ExecutionProgress",
    "FrameExecutionResult",
    "FrameExecutionStatus",
    "GenerationResult",
    "MTPExecutionContext",
    "StreamChunk",
]
