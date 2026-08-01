"""
AliceService - Alice 子系统对外能力门面

提供 run_agent() 和 run_agent_stream() 作为 Agent 计算的稳定入口。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from hivememory.alice.runtime.core import AliceRuntime
from hivememory.core.protocol.models import AgentRunContext, AgentRunResult


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
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AgentRunResult:
        """
        非流式 Agent 计算入口

        给定已准备好的执行上下文，由 Alice 负责调度 Agent runtime 完成一次计算。
        """
        kwargs = {
            "agent_run_context": agent_run_context,
            "generation_options": generation_options,
            "cancel_event": cancel_event,
        }
        if generation_id is not None:
            kwargs["generation_id"] = generation_id
        return await self._runtime.run_agent(**kwargs)

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式 Agent 计算入口

        与 run_agent 相同语义，但以 SSE 事件流方式 yield 结果。
        """
        kwargs = {
            "agent_run_context": agent_run_context,
            "generation_options": generation_options,
            "cancel_event": cancel_event,
        }
        if generation_id is not None:
            kwargs["generation_id"] = generation_id
        async for event in self._runtime.run_agent_stream(**kwargs):
            yield event
