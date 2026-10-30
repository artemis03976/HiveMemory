"""测试专用 CPU — 实现 ``CPUPort`` 的脚本化替身。

任务进程测试以本替身替换 Alice：总线上不注册任何 Alice 路由，进程与入口
代码不为此做任何改动。可配置项覆盖以下场景：交互事件、终态结果、终态前
挂起（取消测试）、取消时返回结果或原样传播、取消后迟迟不收尾（硬取消
测试）与抛出异常。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.workspace.contracts import (
    CPUEventSink,
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUFailureReason,
    CPUInputManifest,
    ExecutionThread,
    OperationEntry,
    OperationSubmitter,
    ThreadOperationSubmitter,
)


@dataclass(frozen=True)
class CPUCall:
    """一次 ``execute`` 调用的入参快照；``thread`` 是进程交来的主线程。"""

    manifest: CPUInputManifest
    generation_options: dict[str, Any] | None
    thread: ExecutionThread


class ScriptedCPU:
    """按脚本推送交互事件并返回终态结果的测试 CPU。

    ``events`` 是返回终态前逐项推送的交互事件；``result`` 是正常结束时
    返回的终态结果；``error`` 在事件推送后抛出；``hang_before_result`` 在
    返回终态前永久挂起，用于停止请求与交付方离开的取消测试。

    收到取消时：配置了 ``cancelled_result`` 就把取消转换为返回值，否则原样
    传播取消（不交回任何结果）。``linger_on_cancel`` 让取消收尾先挂起，直到
    再次被取消，模拟停止后不能及时结束、只能由系统关闭硬取消的 CPU；硬取消
    到达后仍按 ``cancelled_result`` 决定返回还是传播。

    与 Alice 一样，交回的轮次事件以本轮用户消息开头：预置结果没有用户消息
    事件时，以输入清单中的用户消息补上序号 0 的事件。

    调用入参记录在 ``calls``；``finished`` 记录执行是否已经结束（返回、
    抛出或传播取消均算）。
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
        self.finished = False
        #: 挂起点被进入时置位；取消测试据此同步停止请求的注入时机。
        self.hang_entered = asyncio.Event()
        #: 最后一条交互事件即将推送时置位；背压测试据此确认进程停在通道写入上。
        self.last_event_entered = asyncio.Event()
        #: 即将返回配置的终态结果时置位；用于在 CPU 终态之后注入停止。
        self.result_returning = asyncio.Event()

    @property
    def operation_entry(self) -> OperationEntry:
        """测试组合根注入的真实入口，供线程结束后的拒绝断言。"""
        if self._operation_entry is None:
            raise RuntimeError("ScriptedCPU 的操作入口尚未装配")
        return self._operation_entry

    def bind_operation_entry(self, entry: OperationEntry) -> None:
        """以共享入口装配 CPU；每次执行只绑定该次进程的主线程。"""
        self._operation_entry = entry

    async def execute(
        self,
        manifest: CPUInputManifest,
        *,
        thread: ExecutionThread,
        generation_options: dict[str, Any] | None,
        events: CPUEventSink,
    ) -> CPUExecutionResult:
        self.calls.append(
            CPUCall(manifest=manifest, generation_options=generation_options, thread=thread)
        )
        try:
            if self._operation_script is not None:
                await self._operation_script(ThreadOperationSubmitter(self.operation_entry, thread))
            for index, event in enumerate(self._events, start=1):
                if index == len(self._events):
                    self.last_event_entered.set()
                await events(dict(event))
            if self._hang_before_result:
                self.hang_entered.set()
                await asyncio.Event().wait()
            if self._error is not None:
                raise self._error
            if self._result is None:
                raise AssertionError("ScriptedCPU 没有配置终态结果")
            self.result_returning.set()
            return _opened_by_user_message(self._result, manifest)
        except asyncio.CancelledError:
            if self._linger_on_cancel:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()
            if self._cancelled_result is None:
                raise
            return _opened_by_user_message(self._cancelled_result, manifest)
        finally:
            self.finished = True


def _opened_by_user_message(
    result: CPUExecutionResult, manifest: CPUInputManifest
) -> CPUExecutionResult:
    if result.turn_events and result.turn_events[0].kind == "user_message":
        return result
    user = TurnEvent(kind="user_message", sequence=0, content=manifest.user_message)
    return result.model_copy(update={"turn_events": [user, *result.turn_events]})


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
