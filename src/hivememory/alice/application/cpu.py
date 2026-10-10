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
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUInputManifest,
    CPUOutput,
    ExecutionCredential,
    OperationEntry,
    OperationRequest,
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
        credential: ExecutionCredential,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]:
        return self._execute(
            manifest, credential=credential, generation_options=generation_options, stream=stream
        )

    async def _execute(
        self,
        manifest: CPUInputManifest,
        *,
        credential: ExecutionCredential,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]:
        async def submit_operation[R](request: OperationRequest[R]) -> R:
            """绑定本次执行凭据，把操作请求交给 workspace 统一入口。"""
            return await self._operation_entry.execute(request, credential=credential)

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
