"""任务进程的 CPU 端口与 CPU 中立执行结果契约。

端口由 workspace 定义、由 CPU 的提供方实现（当前为 Alice），进程只经
组合根注入的端口调用 CPU，不出现任何具体 CPU 的路由名或结果类型。
本模块只依赖 ``core``，供其他 L3 子系统按"彼此只导入对方 ``contracts``
子包"的规则消费。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.workspace.contracts.process import CPUInputManifest
from hivememory.workspace.contracts.threads import ExecutionThread


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
    """一次 CPU 执行的终态产出（CPU 中立，不含任何执行者专属的字段）。

    字段不变量：每个字段由 CPU 组装，且有完全明确的下游消费者。
        status            → 任务进程按此判定 Actor 阶段结局；有输出的三种结局均进入 finalize
        final_text        → 用户可见回复；轮次事件中没有助手正文时由封口补成事件
        turn_events       → 本轮唯一的交互记录，以序号 0 的用户消息开头 → 封口 → 感知层
        model_used        → 本次 run 实际使用的模型展示名；空字符串表示未解析
        failure_reason    → failed 终态的公开原因码；外部 CPU 可以不填
        compaction_checkpoint → 会话接受的旧历史派生视图，不进入本轮封口或 done

    端口语义：终态结果是 ``execute`` 的返回值；停止后的结局同样以返回值
    表达（``status`` 为 cancelled，携带已产生的输出与已完成的检查点）。
    """

    model_config = ConfigDict(frozen=True)

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


#: 交互事件 sink：CPU 逐项推送带 ``event`` 与 ``data`` 的字典，进程原样
#: 转交流式交付、不做解释。sink 可能因订阅者暂停而等待（背压），停止时
#: 在等待处抛出 ``asyncio.CancelledError``。
type CPUEventSink = Callable[[dict[str, Any]], Awaitable[None]]


class CPUPort(Protocol):
    """任务进程调用 CPU 的对象端口：推送交互事件，返回终态结果。

    ``execute`` 在执行过程中经 ``events`` 推送交互事件，结束时返回唯一的
    终态结果。交付形态由进程决定，CPU 不区分：非流式交付时进程照常接收
    事件后丢弃。端口的取消、异常与参数语义：

    - 进程以原生 ``asyncio.CancelledError`` 取消正在执行的协程（停止请求）。
      CPU 内部各层原样传播取消并完成收尾，只在最外层把取消转换为返回值：
      返回 cancelled 结果，携带已产生的输出与已完成的检查点。没有转换、
      直接传播取消的 CPU 按无输出的取消处理。
    - 收到取消后的收尾不得依赖 actor 操作，只能用已经产生的内容组装结果：
      停止被接受时全部线程已经结束，此后的提交都以线程已结束拒绝。
    - CPU 自报的结局经终态结果的 ``status`` 表达；实现抛出普通异常时由进程
      按失败处理，不接受任何候选结果。
    - ``generation_options`` 由各个 CPU 自行解释，进程原样传递。

    ``thread`` 是本次的主线程：输入清单回答“处理什么”，线程描述回答“谁来
    处理”（标识、观测标签、图纸与操作快照）。CPU 驱动把它绑定为主线程的
    提交函数，操作适配器只经提交函数构造请求，不取得访问 context 或目标
    Workspace。Actor 阶段结束时全部线程同步结束，后续提交明确拒绝。
    """

    async def execute(
        self,
        manifest: CPUInputManifest,
        *,
        thread: ExecutionThread,
        generation_options: dict[str, Any] | None,
        events: CPUEventSink,
    ) -> CPUExecutionResult: ...


__all__ = [
    "CPUEventSink",
    "CPUExecutionResult",
    "CPUExecutionStatus",
    "CPUFailureReason",
    "CPUPort",
]
