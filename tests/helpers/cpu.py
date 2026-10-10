"""测试专用 CPU — 实现 ``CPUPort`` 的脚本化替身。

任务进程测试以本替身替换 Alice：总线上不注册任何 Alice 路由，进程与入口
代码不为此做任何改动。可配置项覆盖计划要求的全部场景：交互事件、终态
结果、拉取时挂起（取消测试）、抛出异常与不产出终态。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    CPUOutput,
    ExecutionCredential,
    OperationEntry,
    OperationRequest,
    OperationSubmitter,
)

_Result = TypeVar("_Result")


@dataclass(frozen=True)
class CPUCall:
    """一次 ``execute`` 调用的入参快照。"""

    manifest: CPUInputManifest
    generation_options: dict[str, Any] | None
    stream: bool
    credential: ExecutionCredential


class ScriptedCPU:
    """按脚本产出 CPU 输出的测试 CPU。

    ``events`` 是终态结果前逐项产出的交互事件；``result`` 是唯一的终态
    执行结果（为 ``None`` 时迭代器不产出终态即结束，模拟协议错误）；
    ``error`` 在事件产出后抛出；``hang_before_result`` 在产出终态前永久
    挂起，用于停止请求与断流的取消测试。

    调用入参记录在 ``calls``；``closed`` 记录输出迭代器是否已终结（被
    ``aclose`` 关闭、被取消终止或自然耗尽均算）。
    """

    def __init__(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        result: CPUExecutionResult | None = None,
        error: Exception | None = None,
        hang_before_result: bool = False,
        operation_script: Callable[[OperationSubmitter], Awaitable[None]] | None = None,
    ) -> None:
        self._events = list(events or [])
        self._result = result
        self._error = error
        self._hang_before_result = hang_before_result
        self._operation_script = operation_script
        self._operation_entry: OperationEntry | None = None
        self.calls: list[CPUCall] = []
        self.closed = False
        #: 挂起点被进入时置位；取消测试据此同步停止请求的注入时机。
        self.hang_entered = asyncio.Event()

    @property
    def operation_entry(self) -> OperationEntry:
        """测试组合根注入的真实入口，供进程关闭后的凭据拒绝断言。"""
        if self._operation_entry is None:
            raise RuntimeError("ScriptedCPU 的操作入口尚未装配")
        return self._operation_entry

    def bind_operation_entry(self, entry: OperationEntry) -> None:
        """以共享入口装配 CPU；每次执行只绑定该次进程的凭据。"""
        self._operation_entry = entry

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        credential: ExecutionCredential,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]:
        self.calls.append(
            CPUCall(
                manifest=manifest,
                generation_options=generation_options,
                stream=stream,
                credential=credential,
            )
        )
        return self._iterate(credential)

    async def _iterate(self, credential: ExecutionCredential) -> AsyncGenerator[CPUOutput, None]:
        try:
            if self._operation_script is not None:

                async def submit(request: OperationRequest[_Result]) -> _Result:
                    return await self.operation_entry.execute(request, credential=credential)

                await self._operation_script(submit)
            for event in self._events:
                yield dict(event)
            if self._hang_before_result:
                self.hang_entered.set()
                await asyncio.Event().wait()
            if self._error is not None:
                raise self._error
            if self._result is not None:
                yield self._result
        finally:
            self.closed = True


def make_cpu_result(
    *,
    status: CPUExecutionStatus | str = CPUExecutionStatus.COMPLETED,
    final_text: str = "完成",
    turn_events: list | None = None,
    model_used: str = "glm-4",
) -> CPUExecutionResult:
    """构造一条默认完成的 CPU 执行结果，字段可覆盖。"""
    return CPUExecutionResult(
        status=status,
        final_text=final_text,
        turn_events=list(turn_events or []),
        model_used=model_used,
    )
