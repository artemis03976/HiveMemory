"""
PassiveObserverIngressor — 旧被动观测模式编排器

该组件已迁入 system/application/passive，作为顶层被动应用层的一部分保留，
便于过渡期测试与剩余兼容链路继续使用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

from hivememory.core.models import Identity
from hivememory.core.protocol.models import InteractionPayload
from hivememory.system.application.passive.models import (
    PassiveIngressEvent,
    PassiveIngressOutcome,
)
from hivememory.system.application.passive.observer_turn_buffer import (
    FlushResult,
    ObserverTurnBufferManager,
)

if TYPE_CHECKING:
    from hivememory.patchouli.eye import TheEye
    from hivememory.core.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)


class PassiveObserverIngressor:
    """旧被动观测模式编排器。"""

    def __init__(self, eye: TheEye, bus: Any = None) -> None:
        self._eye = eye
        self._bus = bus
        self._buffers = ObserverTurnBufferManager()
        self._idle_timeout: float = 30.0
        self._on_flush_callback: Optional[
            Callable[[InteractionPayload, Optional[str]], Coroutine[Any, Any, None]]
        ] = None
        logger.info("PassiveObserverIngressor 初始化完成")

    @property
    def buffers(self) -> ObserverTurnBufferManager:
        return self._buffers

    def configure_idle_flush(
        self,
        timeout_seconds: float = 30.0,
        on_flush_callback: Optional[
            Callable[[InteractionPayload, Optional[str]], Coroutine[Any, Any, None]]
        ] = None,
    ) -> None:
        self._idle_timeout = timeout_seconds
        self._on_flush_callback = on_flush_callback

    def ingest_user(
        self,
        content: str,
        identity: Identity,
    ) -> tuple[EyeGazeResult, Optional[FlushResult]]:
        gaze_result = asyncio.run(
            self._eye.gaze(query=content, topic_snapshots=None, identity=identity)
        )
        buffer = self._buffers.get_buffer(identity)
        flushed = buffer.accept_user(content=content, gaze_result=gaze_result)
        return gaze_result, flushed

    async def ingest_user_async(
        self,
        content: str,
        identity: Identity,
    ) -> tuple[EyeGazeResult, Optional[FlushResult]]:
        gaze_result = await self._eye.gaze(
            query=content, topic_snapshots=None, identity=identity
        )
        buffer = self._buffers.get_buffer(identity)
        flushed = buffer.accept_user(content=content, gaze_result=gaze_result)
        return gaze_result, flushed

    def ingest_assistant(self, content: str, identity: Identity) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_assistant(content)

    def ingest_tool_call(
        self,
        content: str,
        identity: Identity,
        *,
        action_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_kind: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        target: Optional[str] = None,
    ) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_tool_call(
            content,
            action_id=action_id,
            tool_name=tool_name,
            tool_kind=tool_kind,
            tool_args=tool_args,
            target=target,
        )

    def ingest_tool_result(
        self,
        content: str,
        identity: Identity,
        *,
        action_id: Optional[str] = None,
        status: Optional[str] = None,
        render_as: str = "plain",
    ) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_tool_result(
            content,
            action_id=action_id,
            status=status,
            render_as=render_as,
        )

    async def route_event(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
    ) -> PassiveIngressOutcome:
        if event.role == "user":
            gaze_result, flushed = await self.ingest_user_async(
                content=event.content,
                identity=identity,
            )
            return PassiveIngressOutcome(
                kind="user",
                gaze_result=gaze_result,
                flushed=flushed,
            )

        if event.role == "assistant":
            self.ingest_assistant(content=event.content, identity=identity)
            return PassiveIngressOutcome(kind="buffered")

        if event.role == "tool_call":
            self.ingest_tool_call(
                event.content,
                identity,
                action_id=event.action_id,
                tool_name=event.tool_name,
                tool_kind=event.tool_kind,
                tool_args=event.tool_args,
                target=event.target,
            )
            return PassiveIngressOutcome(kind="buffered")

        if event.role == "tool_result":
            self.ingest_tool_result(
                event.content,
                identity,
                action_id=event.action_id,
                status=event.status,
                render_as=event.render_as,
            )
            return PassiveIngressOutcome(kind="buffered")

        return PassiveIngressOutcome(kind="ignored")

    def flush_session(self, identity: Identity) -> Optional[FlushResult]:
        buffer = self._buffers.get_buffer(identity)
        return buffer.flush()

    def flush_idle_sessions(self, timeout_seconds: Optional[float] = None) -> List[FlushResult]:
        timeout = timeout_seconds or self._idle_timeout
        return self._buffers.flush_idle_buffers(timeout)

    def flush_all_pending_sessions(self) -> List[FlushResult]:
        return self._buffers.flush_idle_buffers(-1.0)

    async def scan_idle_sessions_once(self) -> int:
        results = self._buffers.flush_idle_buffers(self._idle_timeout)
        if not results:
            return 0

        for payload, target_topic in results:
            if self._on_flush_callback:
                await self._on_flush_callback(payload, target_topic)

        return len(results)


__all__ = [
    "PassiveObserverIngressor",
]
