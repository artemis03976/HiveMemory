"""Alice 的 CPU 端口：绑定线程操作入口、检查生命周期并管理输出流。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from hivememory.alice.interface.stream import AgentRunStreamAdapter
from hivememory.alice.run.service import AgentRunService
from hivememory.core.access import WorkspaceAccessContext
from hivememory.workspace.contracts import (
    CPUInputManifest,
    CPUOutput,
    CPUResultSlot,
    CPURunHandle,
    OperationEntry,
    ThreadOperationSubmitter,
)


class AliceUnavailableError(RuntimeError):
    """CPU 宿主尚未启动或已经停止，执行阶段不可用。"""

    code = "alice.unavailable"

    def __init__(self) -> None:
        super().__init__("Alice CPU is not running.")
        self.details = {"reason": "not_running"}


class AliceCPU:
    """以 Alice Agent run 充当任务进程 CPU；访问凭据只绑定到提交函数。

    每次执行都经流式适配器运行，逐项交出交互事件再交出终态；交付方只要
    终态时由它丢弃事件，Alice 不区分交付形态。
    """

    def __init__(
        self,
        service: AgentRunService,
        operation_entry: OperationEntry,
        *,
        stream_adapter: AgentRunStreamAdapter,
    ) -> None:
        self._service = service
        self._operation_entry = operation_entry
        self._stream_adapter = stream_adapter
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
    ) -> CPURunHandle:
        slot = CPUResultSlot()
        return CPURunHandle(
            output=self._execute(
                manifest,
                access=access,
                generation_options=generation_options,
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
        result_slot: CPUResultSlot,
    ) -> AsyncGenerator[CPUOutput, None]:
        # 在首次拉取时检查，错误由任务进程的 CPU 执行阶段统一收尾。
        if not self._running:
            raise AliceUnavailableError()
        submit_operation = ThreadOperationSubmitter(self._operation_entry, access)
        adapter = self._stream_adapter.create()
        outputs = adapter.events(
            self._service.run_agent(
                manifest,
                generation_options=generation_options,
                submit_operation=submit_operation,
                run_output=adapter.output,
                result_slot=result_slot,
            )
        )
        # 断流或拉取取消必须关闭内层流，等待 runner 收尾后才交回结果槽。
        async with aclosing(outputs):
            async for output in outputs:
                yield output


__all__ = ["AliceCPU", "AliceUnavailableError"]
