"""Alice Agent run 的 queue-backed 流式输出适配。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.agent_runtime import (
    ExecutionFrame,
    FrameOutput,
    FrameOutputSink,
    MTPFinished,
    MTPStarted,
    TokenDelta,
)
from hivememory.alice.orchestration.output import CallOutputFinished, CallOutputStarted
from hivememory.alice.run.output import RunOutput
from hivememory.components.cancellation import cancel_and_join
from hivememory.workspace.contracts import CPUExecutionResult, CPUOutput

logger = logging.getLogger(__name__)


async def _cancel_runner(task: asyncio.Task[Any]) -> None:
    """取消并等待 runner，同时保留当前消费 task 自己收到的取消。"""
    try:
        await cancel_and_join(task)
    except Exception:
        logger.warning("取消 Agent runner 收尾失败", exc_info=True)


class _BoundFrameOutputSink:
    """为单个 frame 绑定固定元数据的 FrameOutputSink 适配。"""

    def __init__(
        self,
        output: QueueAgentRunOutput,
        metadata: Mapping[str, Any],
    ) -> None:
        self._output = output
        self._metadata = dict(metadata)

    @property
    def streams_tokens(self) -> bool:
        return True

    async def send(self, output: FrameOutput) -> None:
        event, data = _project_frame_output(output)
        await self._output.send_event(event, data, metadata=self._metadata)


@dataclass(frozen=True)
class _PendingProgress:
    """进度交出后等待消费方继续拉取，保留压缩开始与完成时的暂停点。"""

    event: dict[str, Any]
    acknowledged: asyncio.Future[None]


class QueueAgentRunOutput(RunOutput):
    """单个 run 独占的有界交互输出队列。"""

    cancellation_message = "Agent stream closed before terminal event."
    close_reason = "stream_closed"

    def __init__(self, *, maxsize: int = 256) -> None:
        self._agent_run_id = ""
        self._queue: asyncio.Queue[dict[str, Any] | _PendingProgress | None] = asyncio.Queue(
            maxsize=maxsize
        )
        self._sequence = 0

    def bind_run(self, agent_run_id: str) -> None:
        self._agent_run_id = agent_run_id

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None:
        acknowledged: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _PendingProgress(
                self.make_event("context_compaction", {"status": status, **data}), acknowledged
            )
        )
        await acknowledged

    def for_frame(
        self,
        frame: ExecutionFrame,
        *,
        action_id: str | None,
        scope: Literal["main", "sub"],
        depth: int,
    ) -> FrameOutputSink:
        # 展示取 Profile 的源 alias（内置图纸为 omni_doll）；只有未携带身份的 Profile 回退注册标签。
        agent_id = frame.agent_profile.agent_id or frame.runtime_scope.labels.agent_id
        return _BoundFrameOutputSink(
            self,
            metadata={
                "agent_run_id": frame.runtime_scope.run_id,
                "action_id": action_id,
                "scope": scope,
                "depth": depth,
                "agent_id": agent_id,
                "frame_id": frame.runtime_scope.frame_id,
            },
        )

    async def call_started(self, output: CallOutputStarted) -> None:
        await self.send_event(
            "sub_agent_start",
            {
                "agent_id": output.agent_id,
                "task": output.task,
                "iteration": output.iteration,
                "action_id": output.action_id,
                "scope": "sub",
                "depth": 1,
                "frame_id": output.frame_id,
            },
        )

    async def call_finished(self, output: CallOutputFinished) -> None:
        data: dict[str, Any] = {
            "status": output.status,
            "final_text": output.final_text,
            "iteration": output.iteration,
            "action_id": output.action_id,
            "scope": "sub",
            "depth": 1,
            "frame_id": output.frame_id,
            "agent_id": output.agent_id,
        }
        if output.terminal_status is not None:
            data["terminal_status"] = output.terminal_status
        if output.error_code is not None:
            data["error_code"] = output.error_code
        await self.send_event("sub_agent_end", data)

    async def send_event(
        self,
        event: str,
        data: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        await self._queue.put(self.make_event(event, data, metadata=metadata))

    def make_event(
        self,
        event: str,
        data: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """统一编排进度与帧输出的序号。"""
        payload = {
            "agent_run_id": self._agent_run_id,
            **dict(metadata or {}),
            **dict(data),
            "stream_sequence": self._sequence,
        }
        self._sequence += 1
        return {"event": event, "data": payload}

    async def receive(self) -> dict[str, Any] | _PendingProgress | None:
        return await self._queue.get()

    async def finish(self) -> None:
        await self._queue.put(None)


class AgentRunStream:
    """管理一次流式 run 的 runner task、队列消费与断流取消。"""

    def __init__(self, *, queue_size: int = 256) -> None:
        self.output = QueueAgentRunOutput(maxsize=queue_size)

    def events(
        self,
        runner: Awaitable[CPUExecutionResult],
    ) -> AsyncGenerator[CPUOutput, None]:
        async def _events() -> AsyncGenerator[CPUOutput, None]:
            consumer_closed = asyncio.Event()

            async def _run() -> CPUExecutionResult:
                try:
                    return await runner
                except asyncio.CancelledError:
                    raise
                finally:
                    # runner 自身被取消时，仍要唤醒正在消费的客户端；只有消费端已经
                    # 主动关闭时才跳过 sentinel，避免在满队列上等待一个已离开的消费者。
                    if not consumer_closed.is_set():
                        await self.output.finish()

            task = asyncio.create_task(_run())
            try:
                while True:
                    event = await self.output.receive()
                    if event is None:
                        break
                    if isinstance(event, _PendingProgress):
                        yield event.event
                        # 只有继续拉取才允许推进；关闭流时由 finally 取消等待中的 runner。
                        if not event.acknowledged.done():
                            event.acknowledged.set_result(None)
                    else:
                        yield event
                yield await task
            finally:
                consumer_closed.set()
                if not task.done():
                    await _cancel_runner(task)
                elif not task.cancelled():
                    # 若消费端在 runner 异常后、读取 sentinel 前关闭，主动取出异常，
                    # 避免产生 "Task exception was never retrieved"；正常消费路径仍会
                    # 由上面的 await task 把异常传播给调用方。
                    task.exception()

        return _events()


class AgentRunStreamAdapter:
    """为运行用例创建独立的流式输出队列与 runner 管理器。"""

    def __init__(self, *, queue_size: int = 256) -> None:
        self._queue_size = queue_size

    def create(self) -> AgentRunStream:
        return AgentRunStream(queue_size=self._queue_size)


def _project_frame_output(output: FrameOutput) -> tuple[str, dict[str, Any]]:
    """把强类型 frame 输出投影为兼容的交互事件名与数据。"""
    if isinstance(output, TokenDelta):
        return "token", {"content": output.content}
    if isinstance(output, MTPStarted):
        return "mtp_start", {
            "verb": output.verb,
            "target": output.target,
            "args": output.args,
            "raw_text": output.raw_text,
            "iteration": output.iteration,
            "action_id": output.action_id,
        }
    if isinstance(output, MTPFinished):
        return "mtp_result", {
            "verb": output.verb,
            "target": output.target,
            "args": output.args,
            "raw_text": output.raw_text,
            "status": output.status,
            "iteration": output.iteration,
            "action_id": output.action_id,
        }
    raise TypeError(f"Unsupported frame output: {type(output)!r}")


__all__ = [
    "AgentRunStream",
    "AgentRunStreamAdapter",
    "QueueAgentRunOutput",
]
