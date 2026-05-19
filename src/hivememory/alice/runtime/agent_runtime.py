from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from hivememory.core.protocol.models import ChatResult

from hivememory.alice.runtime.loop_executor import KernelLoopExecutor
from hivememory.alice.runtime.worker_agent import WorkerAgentService

if TYPE_CHECKING:
    from hivememory.alice.runtime.frame_scheduler import FrameScheduler
    from hivememory.alice.runtime.runtime import AliceRuntime


class AgentRuntime:
    """Alice 执行态 runtime，持有循环、帧调度与生成引擎。"""

    def __init__(self, runtime: "AliceRuntime") -> None:
        self._runtime = runtime

        from hivememory.alice.runtime.frame_scheduler import FrameScheduler

        self._frame_scheduler = FrameScheduler(runtime=runtime)
        self._worker_agent = WorkerAgentService(config=runtime.config.llm.worker)
        self._loop_executor = KernelLoopExecutor(
            runtime=runtime,
            worker_agent=self._worker_agent,
        )

    @property
    def frame_scheduler(self) -> "FrameScheduler":
        return self._frame_scheduler

    @property
    def loop_executor(self) -> KernelLoopExecutor:
        return self._loop_executor

    @property
    def worker_agent(self) -> WorkerAgentService:
        return self._worker_agent

    async def run_agent(
        self,
        messages: list[dict[str, str]],
        identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> ChatResult:
        return await self._loop_executor.execute_main_frame(
            messages=messages,
            generation_options=generation_options,
            agent_profile=agent_profile,
            topic_id=topic_id,
            identity=identity,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        messages: list[dict[str, str]],
        identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._loop_executor.execute_main_frame_stream(
            messages=messages,
            generation_options=generation_options,
            agent_profile=agent_profile,
            topic_id=topic_id,
            identity=identity,
            cancel_event=cancel_event,
        ):
            yield event

    def health(self) -> dict[str, Any]:
        return {
            "loop_executor": "ok",
            "frame_scheduler": "ok",
            "worker_agent": "ok",
        }


__all__ = ["AgentRuntime"]
