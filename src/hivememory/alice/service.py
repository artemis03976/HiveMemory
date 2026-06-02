"""
AliceService - Alice 子系统对外能力门面

提供 run_agent() 和 run_agent_stream() 作为 Agent 计算的稳定入口。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

from hivememory.core.protocol.models import AgentRunContext, AgentRunResult

from hivememory.alice.runtime.core import AliceRuntime


class AliceService:
    """
    Alice 子系统能力门面

    Phase C 最小接口：
    - run_agent(): 非流式 Agent 计算
    - run_agent_stream(): 流式 Agent 计算
    """

    def __init__(self, runtime: AliceRuntime) -> None:
        self._runtime = runtime

    async def run_agent(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AgentRunResult:
        """
        非流式 Agent 计算入口

        给定已准备好的执行上下文，由 Alice 负责调度 Agent runtime 完成一次计算。
        """
        return await self._runtime.run_agent(
            agent_run_context=agent_run_context,
            generation_options=generation_options,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式 Agent 计算入口

        与 run_agent 相同语义，但以 SSE 事件流方式 yield 结果。
        """
        async for event in self._runtime.run_agent_stream(
            agent_run_context=agent_run_context,
            generation_options=generation_options,
            cancel_event=cancel_event,
        ):
            yield event
