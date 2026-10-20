"""运行用例的输出端口：在调度输出与上下文进度之上补充运行级收尾信息。"""

from __future__ import annotations

from typing import Protocol

from hivememory.alice.context.builder import ContextProgressOutput
from hivememory.alice.orchestration.output import AgentRunOutput


class RunOutput(AgentRunOutput, ContextProgressOutput, Protocol):
    """一次 run 独占的输出端口；传输形态只由实现决定，运行服务不区分。

    ``cancellation_message`` 与 ``close_reason`` 是 run 被取消时
    ``agent.run.cancelled`` 观测事件的载荷取值。
    """

    def bind_run(self, agent_run_id: str) -> None: ...

    @property
    def cancellation_message(self) -> str: ...

    @property
    def close_reason(self) -> str | None: ...


__all__ = ["RunOutput"]
