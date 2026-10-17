"""任务进程的 CPU 端口与 CPU 中立执行结果契约。

端口由 workspace 定义、由 CPU 的提供方实现（当前为 Alice），进程只经
组合根注入的端口调用 CPU，不出现任何具体 CPU 的路由名或结果类型。
本模块只依赖 ``core``，供其他 L3 子系统按"彼此只导入对方 ``contracts``
子包"的规则消费。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.workspace.contracts.process import CPUInputManifest


class CPUExecutionStatus(str, Enum):
    """CPU 自报的单次执行终态。"""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CPUFailureReason(str, Enum):
    """执行者可选上报的稳定失败原因，禁止携带内部异常正文。"""

    INPUT_TOO_LARGE = "input_too_large"
    COMPACTION_FAILED = "compaction_failed"
    TURN_CONTEXT_BUDGET_EXCEEDED = "turn_context_budget_exceeded"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"


class CPUExecutionResult(BaseModel):
    """一次 CPU 执行的终态产出（CPU 中立，取代 Alice 专属的运行结果模型）。

    字段不变量：每个字段由 CPU 组装，且有完全明确的下游消费者。
        status            → 任务进程按此判定 Actor 阶段结局；有输出的三种结局均进入 finalize
        final_text        → 用户可见回复 / InteractionPayload.assistant_final_text
        turn_events       → 封口归约 MTP 轨迹 → 感知层
        model_used        → 本次 run 实际使用的模型展示名；空字符串表示未解析
        failure_reason    → failed 终态的公开原因码；外部 CPU 可以不填
        compaction_checkpoint → 会话接受的旧历史派生视图，不进入本轮封口或 done

    端口语义：终态结果由 ``execute`` 迭代器恰好产出一次且是最后一项；
    非流式时它是唯一一项。迭代器在没有终态结果时结束属于协议错误，
    由进程按失败处理。
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    status: CPUExecutionStatus = Field(default=CPUExecutionStatus.COMPLETED)
    failure_reason: CPUFailureReason | None = Field(default=None)
    compaction_checkpoint: SessionCompactionCheckpoint | None = Field(
        default=None, description="只覆盖已封口旧轮次的新检查点，不算本轮输出"
    )
    final_text: str = Field(default="")
    turn_events: list[TurnEvent] = Field(default_factory=list)
    model_used: str = Field(
        default="",
        description="实际使用的模型展示名，空字符串表示注册表未启用或未解析",
    )

    @property
    def has_output(self) -> bool:
        """初始用户消息是输入；只有回复文本或执行事件才构成本轮输出。"""
        return bool(self.final_text) or any(
            event.kind != "user_message" for event in self.turn_events
        )


#: CPU 输出流的单项：交互事件（带 ``event`` 与 ``data`` 的字典，进程原样
#: 转交流式交付、不做解释）或唯一的终态执行结果。
type CPUOutput = dict[str, Any] | CPUExecutionResult


@dataclass
class CPUResultSlot:
    """CPU 同步写入的终态结果槽；取消传播时仍可交回已产生的输出，结局如实填写、进程不改写。"""

    result: CPUExecutionResult | None = None


@dataclass(frozen=True)
class CPURunHandle:
    """一次 CPU 执行的输出流与结果槽；不暴露具体执行器或进程控制。"""

    output: AsyncGenerator[CPUOutput, None]
    result_slot: CPUResultSlot


class CPUPort(Protocol):
    """任务进程调用 CPU 的对象端口。

    ``execute`` 是以 ``stream`` 参数控制是否流式的统一入口，返回一个异步
    run 句柄：输出流先产出交互事件、再产出唯一终态；非流式只产出终态。
    取消收尾时 CPU 同步写入结果槽，再传播原生 CancelledError。进程在关闭流程中
    以 ``aclose`` 关闭它（含断流与取消路径）。端口的取消、异常与关闭语义：

    - 进程取消正在拉取下一项的任务（用户停止），并在关闭流程中关闭迭代器；
      实现必须传播 ``asyncio.CancelledError``，并在迭代器被关闭时释放自己
      创建的资源。
    - CPU 自报的结局经终态结果的 ``status`` 表达；实现抛出异常时由进程
      按失败处理。
    - ``generation_options`` 由各个 CPU 自行解释，进程原样传递。

    ``access`` 是本次主线程的密封线程 context，独立于冻结输入清单。
    CPU 驱动将它绑定到操作提交函数；操作适配器只构造请求，不取得访问
    context 或目标 Workspace。CPU 执行结束后同步撤销，后续提交明确拒绝。
    """

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> CPURunHandle: ...


__all__ = [
    "CPUFailureReason",
    "CPURunHandle",
    "CPUResultSlot",
    "CPUExecutionResult",
    "CPUExecutionStatus",
    "CPUPort",
    "CPUOutput",
]
