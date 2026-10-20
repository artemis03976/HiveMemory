"""单帧执行的状态、策略、观测坐标与停机信号。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from hivememory.core.models import AgentProfile, ExecutionLabels, TurnEvent
from hivememory.core.mtp.models import MTPCallRequest

if TYPE_CHECKING:
    from hivememory.agent_runtime.model.resolution import ResolvedFrameModel
    from hivememory.workspace.contracts import OperationSubmitter


class RuntimeScope(BaseModel):
    """单次 run/frame/action 坐标与所属执行线程的观测标签。"""

    labels: ExecutionLabels
    run_id: str
    frame_id: str
    action_id: str | None = None

    def with_action(self, action_id: str) -> RuntimeScope:
        """返回限定在单个 agent 动作范围内的副本。"""
        return self.model_copy(update={"action_id": action_id})

    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True)
class FrameExecutionPolicy:
    """单次 frame 可执行的 MTP 动词集合与迭代预算，不承担资源授权。"""

    permitted_verbs: frozenset[str]
    max_iterations: int | None = None

    @classmethod
    def from_verbs(
        cls,
        permitted_verbs: Iterable[str],
        *,
        max_iterations: int | None = None,
        denied_verbs: Iterable[str] = (),
    ) -> FrameExecutionPolicy:
        """从明确的动词集合构造策略，并可选移除子 frame 禁止的动词。"""
        permitted = frozenset(verb.upper() for verb in permitted_verbs)
        denied = {verb.upper() for verb in denied_verbs}
        return cls(permitted_verbs=permitted.difference(denied), max_iterations=max_iterations)

    def allows(self, verb: str) -> bool:
        """只允许集合中的动词；大小写不敏感，未知动词不会默认放行。"""
        return verb.upper() in self.permitted_verbs


@dataclass(frozen=True)
class FrameProducts:
    """从一次成功收尾的 frame 投影出的产物（仅供当前 CALL 使用）。"""

    artifact_aliases: tuple[str, ...] = ()


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
    保存每帧的观测标签与执行策略。资源访问由 CPU 驱动绑定到所属线程的提交函数负责，
    标签仅用于展示与观测。执行进度（正文、事件、迭代与序号）
    全部放在 ``progress`` 上，CALL 挂起后重入同一 frame 可自然续接。
    """

    runtime_scope: RuntimeScope
    agent_profile: AgentProfile
    working_history: list[dict[str, str]]
    topic_id: str | None
    execution_policy: FrameExecutionPolicy

    # 每帧使用所属线程的提交函数；线程 context 由 CPU 驱动绑定。
    submit_operation: OperationSubmitter
    # 本帧的已解析语言：MTP 回填、错误与 CALL 响应沿用它，与提示词保持一致，
    # 不在执行中从 Profile 或进程默认值重新推导。
    language: str

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

    本帧累积产物保存在 ``frame.progress``（``ExecutionProgress``）中。这里只表达
    "为什么停下来"，以及挂起时编排派生子帧所需的最小信息。

    引擎语义：``execute_frame(frame)`` 读写传入的 ``frame``，跑到自然收敛返回
    ``COMPLETED``，命中 CALL 返回 ``SUSPENDED`` 并把控制权交还编排；失败和循环预算
    耗尽分别返回对应终态，原生任务取消直接沿调用栈传播。
    引擎自己不 fork、不 resume、不组 CALL response。
    CPU 执行结果由编排在 frame 终态确定后从 ``frame.progress`` 聚合。
    """

    status: FrameExecutionStatus

    # ---- status == SUSPENDED 时填充 ----
    # 触发 CALL 的派生请求（target_alias / task / context_refs）。
    call_request: MTPCallRequest | None = None
    # MTP 生成器已为被 stop sequence 截断的挂起文本补全右定界符。
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
