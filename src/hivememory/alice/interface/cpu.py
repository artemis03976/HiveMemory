"""``CPUPort`` 的 Alice 实现 — 任务进程经端口调用 Alice Agent run。

端口对象由组合根注入任务进程；进程只持有端口，Alice 的运行时仍在公开
路由之后。本实现经全局总线选择 Alice 的流式或非流式执行路由（见
``AgentRunService.run_agent``）：流式时交互事件原样透传、``done`` 事件被
转换为 CPU 中立的执行结果（``done`` 中的运行元数据不进入结果，进程同样
丢弃它们）；非流式时路由直接返回执行结果。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from hivememory.alice.contracts import AliceRoutes
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUInputManifest,
    CPUOutput,
    CPUResultSlot,
    CPURunHandle,
    OperationEntry,
    ThreadOperationSubmitter,
)


class AliceCPU:
    """以 Alice Agent run 充当任务进程 CPU 的端口实现。"""

    def __init__(self, global_bus: GlobalSystemBus, operation_entry: OperationEntry) -> None:
        self._bus = global_bus
        self._operation_entry = operation_entry

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> CPURunHandle:
        slot = CPUResultSlot()
        return CPURunHandle(
            output=self._execute(
                manifest,
                access=access,
                generation_options=generation_options,
                stream=stream,
                result_slot=slot,
            ),
            result_slot=slot,
        )

    async def _execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        stream: bool,
        result_slot: CPUResultSlot,
    ) -> AsyncGenerator[CPUOutput, None]:
        submit_operation = ThreadOperationSubmitter(self._operation_entry, access)

        if stream:
            event_stream = await self._bus.request(
                AliceRoutes.RUN_AGENT_STREAM,
                input_manifest=manifest,
                submit_operation=submit_operation,
                result_slot=result_slot,
                generation_options=generation_options,
            )
            # 无论耗尽、被进程关闭还是在拉取中被取消，都要把 Alice 的事件流
            # 一并关闭，使 Alice 以取消语义收口自己的 run。
            async with aclosing(event_stream):
                async for event in event_stream:
                    if event["event"] == "done":
                        # done 除结果字段外还携带运行元数据；构造执行结果时
                        # 按模型校验还原 turn_events 等字段，多余字段被忽略。
                        result = CPUExecutionResult(**event["data"])
                        # 检查点只经结果槽交给进程，不进入交互事件载荷。
                        if result_slot.result is not None:
                            result = result.model_copy(
                                update={
                                    "compaction_checkpoint": result_slot.result.compaction_checkpoint
                                }
                            )
                        yield result
                    else:
                        yield event
        else:
            result = await self._bus.request(
                AliceRoutes.RUN_AGENT,
                input_manifest=manifest,
                submit_operation=submit_operation,
                result_slot=result_slot,
                generation_options=generation_options,
            )
            yield result


__all__ = ["AliceCPU"]
