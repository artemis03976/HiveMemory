"""测试专用 CPU — 实现 ``CPUPort`` 的脚本化替身。

任务进程测试以本替身替换 Alice：总线上不注册任何 Alice 路由，进程与入口
代码不为此做任何改动。可配置项覆盖以下场景：交互事件、终态
结果、终态前挂起（取消测试）、取消后迟迟不收尾（硬取消测试）、抛出异常与
不产出终态。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUFailureReason,
    CPUInputManifest,
    CPUOutput,
    CPUResultSlot,
    CPURunHandle,
    OperationEntry,
    OperationSubmitter,
    ThreadOperationSubmitter,
)


@dataclass(frozen=True)
class CPUCall:
    """一次 ``execute`` 调用的入参快照。"""

    manifest: CPUInputManifest
    generation_options: dict[str, Any] | None
    access: WorkspaceAccessContext


class ScriptedCPU:
    """按脚本产出 CPU 输出的测试 CPU。

    ``events`` 是终态结果前逐项产出的交互事件；``result`` 是唯一的终态
    执行结果（为 ``None`` 时迭代器不产出终态即结束，模拟协议错误）；
    ``error`` 在事件产出后抛出；``hang_before_result`` 在产出终态前永久
    挂起，用于停止请求与交付方离开的取消测试；``linger_on_cancel`` 让取消
    收尾在回填结果槽后继续挂起，直到再次被取消，模拟停止后不能及时结束、
    只能由系统关闭硬取消的 CPU。

    调用入参记录在 ``calls``；``closed`` 记录输出迭代器是否已终结（被
    ``aclose`` 关闭、被取消终止或自然耗尽均算）。
    """

    def __init__(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        result: CPUExecutionResult | None = None,
        error: Exception | None = None,
        cancelled_result: CPUExecutionResult | None = None,
        hang_before_result: bool = False,
        linger_on_cancel: bool = False,
        operation_script: Callable[[OperationSubmitter], Awaitable[None]] | None = None,
    ) -> None:
        self._events = list(events or [])
        self._result = result
        self._error = error
        self._cancelled_result = cancelled_result
        self._hang_before_result = hang_before_result
        self._linger_on_cancel = linger_on_cancel
        self._operation_script = operation_script
        self._operation_entry: OperationEntry | None = None
        self.calls: list[CPUCall] = []
        self.closed = False
        #: 挂起点被进入时置位；取消测试据此同步停止请求的注入时机。
        self.hang_entered = asyncio.Event()
        #: 最后一条交互事件即将交出时置位；背压测试据此确认进程停在通道写入上。
        self.last_event_entered = asyncio.Event()

    @property
    def operation_entry(self) -> OperationEntry:
        """测试组合根注入的真实入口，供线程 context 撤销后的拒绝断言。"""
        if self._operation_entry is None:
            raise RuntimeError("ScriptedCPU 的操作入口尚未装配")
        return self._operation_entry

    def bind_operation_entry(self, entry: OperationEntry) -> None:
        """以共享入口装配 CPU；每次执行只绑定该次进程的线程 context。"""
        self._operation_entry = entry

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
    ) -> CPURunHandle:
        self.calls.append(
            CPUCall(manifest=manifest, generation_options=generation_options, access=access)
        )
        slot = CPUResultSlot()
        return CPURunHandle(output=self._iterate(access, slot), result_slot=slot)

    async def _iterate(
        self, access: WorkspaceAccessContext, slot: CPUResultSlot
    ) -> AsyncGenerator[CPUOutput, None]:
        try:
            if self._operation_script is not None:
                await self._operation_script(ThreadOperationSubmitter(self.operation_entry, access))
            for index, event in enumerate(self._events, start=1):
                if index == len(self._events):
                    self.last_event_entered.set()
                yield dict(event)
            if self._hang_before_result:
                self.hang_entered.set()
                await asyncio.Event().wait()
            if self._error is not None:
                raise self._error
            if self._result is not None:
                slot.result = self._result
                yield self._result
        except (asyncio.CancelledError, GeneratorExit):
            slot.result = self._cancelled_result
            if self._linger_on_cancel:
                await asyncio.Event().wait()
            raise
        finally:
            self.closed = True


def make_cpu_result(
    *,
    status: CPUExecutionStatus | str = CPUExecutionStatus.COMPLETED,
    final_text: str = "完成",
    turn_events: list | None = None,
    model_used: str = "glm-4",
    failure_reason: CPUFailureReason | None = None,
    compaction_checkpoint: SessionCompactionCheckpoint | None = None,
) -> CPUExecutionResult:
    """构造一条默认完成的 CPU 执行结果，字段可覆盖。"""
    return CPUExecutionResult(
        status=status,
        final_text=final_text,
        turn_events=list(turn_events or []),
        model_used=model_used,
        failure_reason=failure_reason,
        compaction_checkpoint=compaction_checkpoint,
    )
