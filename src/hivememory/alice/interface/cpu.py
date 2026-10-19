"""Alice 的 CPU 端口：绑定线程操作入口、检查生命周期并把交互输出接到进程的事件 sink。"""

from __future__ import annotations

from typing import Any

from hivememory.alice.interface.output import SinkRunOutput
from hivememory.alice.run.service import AgentRunService
from hivememory.core.access import WorkspaceAccessContext
from hivememory.workspace.contracts import (
    CPUEventSink,
    CPUExecutionResult,
    CPUInputManifest,
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

    每次执行直接等待运行服务：交互事件经输出端口推送给进程的 sink，终态
    （含停止后的 cancelled 结果）作为返回值交回；交付方只要终态时由进程
    丢弃事件，Alice 不区分交付形态。
    """

    def __init__(self, service: AgentRunService, operation_entry: OperationEntry) -> None:
        self._service = service
        self._operation_entry = operation_entry
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        events: CPUEventSink,
    ) -> CPUExecutionResult:
        # 在首次 await 时检查，错误由任务进程的 Actor 阶段按失败收尾。
        if not self._running:
            raise AliceUnavailableError()
        return await self._service.run_agent(
            manifest,
            generation_options=generation_options,
            submit_operation=ThreadOperationSubmitter(self._operation_entry, access),
            run_output=SinkRunOutput(events),
        )


__all__ = ["AliceCPU", "AliceUnavailableError"]
