"""运行用例的输出端口：在调度输出与上下文进度之上补充运行级收尾信息。"""

from __future__ import annotations

from typing import Protocol

from hivememory.alice.context.builder import ContextProgressOutput
from hivememory.alice.orchestration.output import AgentRunOutput, NullAgentRunOutput


class RunOutput(AgentRunOutput, ContextProgressOutput, Protocol):
    """一次 run 独占的输出端口；传输形态只由实现决定，运行服务不区分。"""

    def bind_run(self, agent_run_id: str) -> None: ...

    @property
    def cancellation_message(self) -> str: ...

    @property
    def close_reason(self) -> str | None: ...


class NullRunOutput(NullAgentRunOutput):
    """非流式 run 的空输出：不交付交互事件，取消观测不带关闭原因。"""

    cancellation_message = "Agent run cancelled before terminal event."
    close_reason = None

    def bind_run(self, agent_run_id: str) -> None:
        del agent_run_id

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None:
        del status, data


__all__ = ["NullRunOutput", "RunOutput"]
