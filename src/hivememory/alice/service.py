"""
AliceService - Alice 子系统对外能力门面

提供 run_agent() 和 run_agent_stream() 作为 Agent 计算的稳定入口。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from hivememory.core.models import Identity, MemoryAtom
from hivememory.core.protocol.models import ChatResult

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
        messages: List[Dict[str, str]],
        identity: Identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ChatResult:
        """
        非流式 Agent 计算入口

        给定已准备好的执行上下文，由 Alice 负责调度 Agent runtime 完成一次计算。
        """
        return await self._runtime.run_agent(
            messages=messages,
            identity=identity,
            agent_id=agent_id,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        messages: List[Dict[str, str]],
        identity: Identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式 Agent 计算入口

        与 run_agent 相同语义，但以 SSE 事件流方式 yield 结果。
        """
        async for event in self._runtime.run_agent_stream(
            messages=messages,
            identity=identity,
            agent_id=agent_id,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        ):
            yield event

    async def register_preretrieval_aliases(
        self,
        memories: List[MemoryAtom],
    ) -> None:
        """将预检索命中的记忆别名注入 Alice 运行时缓存。"""
        self._runtime.register_preretrieval_aliases(memories)

    async def get_interaction_state(self) -> Dict[str, Any]:
        """导出当前一轮 Agent 运行积累的 MTP 交互状态。"""
        return self._runtime.export_interaction_state()
