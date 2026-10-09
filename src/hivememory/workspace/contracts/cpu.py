"""任务进程的 CPU 端口与 CPU 中立执行结果契约。

端口由 workspace 定义、由 CPU 的提供方实现（当前为 Alice），进程只经
组合根注入的端口调用 CPU，不出现任何具体 CPU 的路由名或结果类型。
本模块只依赖 ``core``，供其他 L3 子系统按"彼此只导入对方 ``contracts``
子包"的规则消费。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import TurnEvent
from hivememory.workspace.contracts.operations import ProcessOperations
from hivememory.workspace.contracts.process import CPUInputManifest


class CPUExecutionStatus(str, Enum):
    """CPU 自报的单次执行终态。"""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CPUExecutionResult(BaseModel):
    """一次 CPU 执行的终态产出（CPU 中立，取代 Alice 专属的运行结果模型）。

    字段不变量：每个字段由 CPU 组装，且有完全明确的下游消费者。
        status            → 任务进程按此判定 Actor 阶段结局；仅 completed 进入 finalize
        final_text        → 用户可见回复 / InteractionPayload.assistant_final_text
        turn_events       → 封口归约 MTP 轨迹 → 感知层
        model_used        → 本次 run 实际使用的模型展示名；空字符串表示未解析

    端口语义：终态结果由 ``execute`` 迭代器恰好产出一次且是最后一项；
    非流式时它是唯一一项。迭代器在没有终态结果时结束属于协议错误，
    由进程按失败处理。
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    status: CPUExecutionStatus = Field(default=CPUExecutionStatus.COMPLETED)
    final_text: str = Field(default="")
    turn_events: list[TurnEvent] = Field(default_factory=list)
    model_used: str = Field(
        default="",
        description="实际使用的模型展示名，空字符串表示注册表未启用或未解析",
    )


#: CPU 输出流的单项：交互事件（带 ``event`` 与 ``data`` 的字典，进程原样
#: 转交流式交付、不做解释）或唯一的终态执行结果。
type CPUOutput = dict[str, Any] | CPUExecutionResult


class CPUPort(Protocol):
    """任务进程调用 CPU 的对象端口。

    ``execute`` 是以 ``stream`` 参数控制是否流式的统一入口，返回一个异步
    生成器：流式时先产出交互事件、再产出唯一的终态结果；非流式时只产出
    终态结果。返回 ``AsyncGenerator`` 是契约的一部分——进程在关闭流程中
    以 ``aclose`` 关闭它（含断流与取消路径）。端口的取消、异常与关闭语义：

    - 进程取消正在拉取下一项的任务（用户停止），并在关闭流程中关闭迭代器；
      实现必须传播 ``asyncio.CancelledError``，并在迭代器被关闭时释放自己
      创建的资源。
    - CPU 自报的结局经终态结果的 ``status`` 表达；实现抛出异常时由进程
      按失败处理。
    - ``generation_options`` 由各个 CPU 自行解释，进程原样传递。
    """

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        operations: ProcessOperations,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]: ...


__all__ = [
    "CPUExecutionResult",
    "CPUExecutionStatus",
    "CPUPort",
    "CPUOutput",
]
