"""``CPUPort`` 的 Alice 实现 — 任务进程经端口调用 Alice Agent run。

端口对象由组合根注入任务进程；进程只持有端口，Alice 的运行时仍在公开
路由之后。本实现经全局总线调用 Alice 的统一执行路由（见
``AgentRunService.run_agent``）：流式时交互事件原样透传、``done`` 事件被
转换为 CPU 中立的执行结果（``done`` 中的运行元数据不进入结果，进程同样
丢弃它们）；非流式时路由直接返回执行结果。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.workspace.contracts import (
    CallThread,
    CPUExecutionResult,
    CPUInputManifest,
    CPUOutput,
    DeriveCallThreadRequest,
    OperationEntry,
    OperationRequest,
)


class ThreadOperationSubmitter:
    """CPU 驱动绑定的线程提交函数；凭据不交给 frame、MTP 或 CALL 编排。"""

    def __init__(self, entry: OperationEntry, access: WorkspaceAccessContext) -> None:
        self._entry = entry
        self._access = access

    async def __call__[R](self, request: OperationRequest[R]) -> R:
        """普通请求透传；派生结果含凭据，只能通过专用方法转换后交付。"""
        if isinstance(request, DeriveCallThreadRequest):
            raise TypeError("派生请求必须通过 derive_call_thread 提交")
        return await self._entry.execute(request, access=self._access)

    async def derive_call_thread(self, request: DeriveCallThreadRequest) -> CallThread:
        """以本线程提交 CALL，绑定子线程后只交付运行层可用的数据。"""
        child = await self._entry.execute(request, access=self._access)
        return CallThread(
            agent_profile=child.agent_profile,
            allowed_operations=child.allowed_operations,
            submit_operation=ThreadOperationSubmitter(self._entry, child.access),
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
    ) -> AsyncGenerator[CPUOutput, None]:
        return self._execute(
            manifest, access=access, generation_options=generation_options, stream=stream
        )

    async def _execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]:
        submit_operation = ThreadOperationSubmitter(self._operation_entry, access)

        if stream:
            event_stream = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT,
                input_manifest=manifest,
                submit_operation=submit_operation,
                generation_options=generation_options,
                stream=True,
            )
            # 无论耗尽、被进程关闭还是在拉取中被取消，都要把 Alice 的事件流
            # 一并关闭，使 Alice 以取消语义收口自己的 run。
            async with aclosing(event_stream):
                async for event in event_stream:
                    if event["event"] == "done":
                        # done 除结果字段外还携带运行元数据；构造执行结果时
                        # 按模型校验还原 turn_events 等字段，多余字段被忽略。
                        yield CPUExecutionResult(**event["data"])
                    else:
                        yield event
        else:
            yield await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT,
                input_manifest=manifest,
                submit_operation=submit_operation,
                generation_options=generation_options,
                stream=False,
            )


__all__ = ["AliceCPU"]
