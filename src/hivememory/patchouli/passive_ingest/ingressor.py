"""
PassiveObserverIngressor — 被动观测模式编排器

从 TheEye 中独立出来的被动 ingest 编排器，负责：
    - 接收外部离散事件 (user / assistant / tool_call / tool_result)
    - 调用 TheEye.gaze() 为新一轮 user 建立 route 元数据
    - 管理多个 session 的 ObserverTurnBuffer
    - 执行 Next User Turn flush / Idle Timeout flush / Explicit EOF flush
    - 产出结构化 (payload, target_topic) 对

TheEye 不再持有 ObserverBufferManager，只保留 gaze() 职责。

作者: HiveMemory Team
版本: 2.0.0 (Phase P2 — 结构化事件缓冲)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.passive_ingest.models import (
    PassiveIngressEvent,
    PassiveIngressOutcome,
)
from hivememory.patchouli.passive_ingest.observer_turn_buffer import (
    ObserverTurnBuffer,
    ObserverTurnBufferManager,
    FlushResult,
)
from hivememory.patchouli.protocol.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.patchouli.eye import TheEye
    from hivememory.patchouli.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)


class PassiveObserverIngressor:
    """
    被动观测模式编排器

    接管原 TheEye 中 observer session 管理的全部职责：
    - buffer 池管理
    - ingest user/assistant
    - flush (Next User Turn / Idle Timeout / Explicit EOF)
    - idle monitor 调度
    """

    def __init__(
        self,
        eye: TheEye,
        bus: Any = None,
    ) -> None:
        self._eye = eye
        self._bus = bus
        self._buffers = ObserverTurnBufferManager()

        # idle monitor 状态
        self._idle_scheduler = None
        self._idle_monitor_enabled: bool = False
        self._scan_interval_seconds: float = 10.0
        self._monitor_idle_shutdown_seconds: float = 60.0
        self._last_message_ts: Optional[float] = None
        self._idle_timeout: float = 30.0
        self._on_flush_callback = None

        logger.info("PassiveObserverIngressor 初始化完成")

    @property
    def buffers(self) -> ObserverTurnBufferManager:
        return self._buffers

    # ========== 事件接收 ==========

    def ingest_user(
        self,
        content: str,
        identity: Identity,
    ) -> tuple[EyeGazeResult, Optional[FlushResult]]:
        """
        被动模式: 接收 user 消息（同步版本）

        Returns:
            (gaze_result, flushed):
                gaze_result: Eye 分析结果
                flushed: 若触发 "Next User Turn" flush，返回 (上一轮 payload, 上一轮 target_topic)
        """
        gaze_result = asyncio.run(
            self._eye.gaze(query=content, topic_snapshots=None, identity=identity)
        )
        self._on_message_received()
        buffer = self._buffers.get_buffer(identity)
        flushed = buffer.accept_user(content=content, gaze_result=gaze_result)
        return gaze_result, flushed

    async def ingest_user_async(
        self,
        content: str,
        identity: Identity,
    ) -> tuple[EyeGazeResult, Optional[FlushResult]]:
        """
        被动模式: 接收 user 消息（异步版本）

        Returns:
            (gaze_result, flushed):
                gaze_result: Eye 分析结果
                flushed: 若触发 "Next User Turn" flush，返回 (上一轮 payload, 上一轮 target_topic)
        """
        gaze_result = await self._eye.gaze(
            query=content, topic_snapshots=None, identity=identity
        )
        self._on_message_received()
        buffer = self._buffers.get_buffer(identity)
        flushed = buffer.accept_user(content=content, gaze_result=gaze_result)
        return gaze_result, flushed

    def ingest_assistant(
        self,
        content: str,
        identity: Identity,
    ) -> None:
        """被动模式: 接收 assistant 消息，缓冲等待 flush"""
        self._on_message_received()
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
        """被动模式: 接收 tool_call 事件，缓冲等待 flush"""
        self._on_message_received()
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
        """被动模式: 接收 tool_result 事件，缓冲等待 flush"""
        self._on_message_received()
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
        """
        将统一被动事件路由到对应 ingest 分支。

        该方法只负责被动入口侧的事件分发与缓冲结果归一化，
        不负责 kernel 提交、hot 路由或外部 API 返回值组织。
        """
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
            self.ingest_assistant(
                content=event.content,
                identity=identity,
            )
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

    # ========== Flush ==========

    def flush_session(self, identity: Identity) -> Optional[FlushResult]:
        """显式 flush 指定 session (Explicit EOF 触发器)"""
        buffer = self._buffers.get_buffer(identity)
        return buffer.flush()

    def flush_idle_sessions(
        self, timeout_seconds: Optional[float] = None
    ) -> List[FlushResult]:
        """扫描并 flush 所有超时的 buffer (Idle Timeout 触发器)"""
        timeout = timeout_seconds or self._idle_timeout
        return self._buffers.flush_idle_buffers(timeout)

    def flush_all_pending_sessions(self) -> List[FlushResult]:
        """强制 flush 所有仍有 pending round 的 buffer"""
        return self._buffers.flush_idle_buffers(-1.0)

    # ========== Idle Monitor ==========

    def start_idle_monitor(
        self,
        timeout_seconds: float = 30.0,
        scan_interval_seconds: float = 10.0,
        idle_shutdown_seconds: Optional[float] = None,
        lazy_start: bool = False,
        on_flush_callback=None,
    ) -> None:
        if self._idle_scheduler is not None:
            logger.warning("Observer idle monitor 已在运行")
            return

        self._idle_timeout = timeout_seconds
        self._scan_interval_seconds = scan_interval_seconds
        self._monitor_idle_shutdown_seconds = (
            idle_shutdown_seconds
            if idle_shutdown_seconds is not None
            else max(timeout_seconds, scan_interval_seconds * 3)
        )
        self._on_flush_callback = on_flush_callback
        self._idle_monitor_enabled = True
        if lazy_start:
            logger.info(
                f"Observer idle monitor 已配置为惰性启动: "
                f"timeout={timeout_seconds}s, interval={scan_interval_seconds}s, "
                f"idle_shutdown={self._monitor_idle_shutdown_seconds}s"
            )
            return
        self._ensure_idle_monitor_started()

    def stop_idle_monitor(self) -> None:
        if self._idle_scheduler is not None:
            self._idle_scheduler.shutdown(wait=False)
            self._idle_scheduler = None
            logger.info("Observer idle monitor 已停止")

    def _ensure_idle_monitor_started(self) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

        if self._idle_scheduler is not None:
            return
        self._idle_scheduler = BackgroundScheduler()
        self._idle_scheduler.add_job(
            self._scan_idle_buffers,
            "interval",
            seconds=self._scan_interval_seconds,
        )
        self._idle_scheduler.add_listener(
            self._on_scheduler_event,
            EVENT_JOB_ERROR | EVENT_JOB_MISSED,
        )
        self._idle_scheduler.start()
        logger.info(
            f"Observer idle monitor 已启动: "
            f"timeout={self._idle_timeout}s, "
            f"interval={self._scan_interval_seconds}s, "
            f"idle_shutdown={self._monitor_idle_shutdown_seconds}s"
        )

    def _on_scheduler_event(self, event) -> None:
        if getattr(event, "exception", None):
            logger.error(
                f"Observer idle monitor 任务异常: "
                f"job_id={getattr(event, 'job_id', 'unknown')}, "
                f"error={event.exception}, "
                f"traceback={getattr(event, 'traceback', '')}"
            )
            return
        logger.warning(
            f"Observer idle monitor 错过调度: "
            f"job_id={getattr(event, 'job_id', 'unknown')}"
        )

    def _on_message_received(self) -> None:
        self._last_message_ts = time.time()
        if self._idle_monitor_enabled:
            self._ensure_idle_monitor_started()

    def _dispatch_flush_results(self, results: List[FlushResult]) -> None:
        if not results:
            return
        if self._bus:
            for payload, target_topic in results:
                self._bus.emit(
                    "observer.idle_flushed",
                    payload=payload,
                    target_topic=target_topic,
                )
            return
        if self._on_flush_callback:
            for payload, target_topic in results:
                self._on_flush_callback(payload, target_topic)

    def _scan_idle_buffers(self) -> None:
        try:
            results = self._buffers.flush_idle_buffers(self._idle_timeout)
            self._dispatch_flush_results(results)
            if self._last_message_ts is None:
                return
            idle_duration = time.time() - self._last_message_ts
            if idle_duration < self._monitor_idle_shutdown_seconds:
                return
            final_results = self._buffers.flush_idle_buffers(-1.0)
            self._dispatch_flush_results(final_results)
            self.stop_idle_monitor()
            logger.info(
                f"Observer idle monitor 自动关闭: idle={idle_duration:.1f}s"
            )
        except Exception as e:
            logger.error(f"Observer idle monitor 扫描失败: {e}", exc_info=True)


__all__ = [
    "PassiveObserverIngressor",
]
