"""
帕秋莉·真理之眼 (The Eye of Patchouli)

定位：Agentic Dispatcher（进程调度员）与感知者
职责：
    - 流量入口和意图判断 (Hot)
    - 读取活跃话题列表，执行 Agentic Routing（话题路由）
    - 调用 GatewayEngine 获取原始结果（含 target_topic）
    - 处理 fallback、日志、数据转换
    - 产出 EyeGazeResult 供 Kernel 进行数据格式转换
    - 被动观测模式下管理 ObserverBuffer 池 (Passive Observer Mode)

作者: HiveMemory Team
版本: 3.0 (Phase 4.5 Agentic Dispatcher)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, List, Optional

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import InteractionPayload

from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
)
from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.observer_buffer import (
    ObserverBufferManager,
    ObserverSessionBuffer,
)

# 导入协议消息
from hivememory.patchouli.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)

class TheEye:
    def __init__(
        self,
        engine: GatewayEngine,
        bus=None,
    ):
        """
        初始化真理之眼 (Agentic Dispatcher)

        Args:
            engine: Gateway 引擎实例
            bus: SystemBus 实例，用于跨服务通信（替代 perception_layer 直接引用）
        """
        self._engine = engine
        self._bus = bus
        self._observer_buffers = ObserverBufferManager()
        self._observer_idle_scheduler = None
        self._observer_idle_timeout: float = 30.0

        logger.info(f"TheEye 真理之眼初始化完成 (Agentic Dispatcher)")

    # ========== MMU 话题菜单 (Phase 4.5) ==========

    def _build_active_topics_menu(self) -> Optional[str]:
        """
        从感知层获取活跃话题菜单，格式化为 Dispatcher prompt 可用的字符串

        Returns:
            格式化的话题菜单字符串，无话题时返回 None
        """
        if self._bus is None:
            return None

        try:
            menu = self._bus.request("librarian.get_active_topics_menu")
            if not menu:
                return None

            lines = []
            for item in menu:
                lines.append(f'"{item["topic_id"]}: {item["title"]}"')
            return "[" + ", ".join(lines) + "]"
        except Exception as e:
            logger.warning(f"获取活跃话题菜单失败: {e}")
            return None

    async def _build_active_topics_menu_async(self) -> Optional[str]:
        if self._bus is None:
            return None
        try:
            menu = await self._bus.async_request(
                "librarian.get_active_topics_menu"
            )
            if not menu:
                return None
            lines = []
            for item in menu:
                lines.append(f'"{item["topic_id"]}: {item["title"]}"')
            return "[" + ", ".join(lines) + "]"
        except Exception as e:
            logger.warning(f"获取活跃话题菜单失败: {e}")
            return None

    async def gaze(
        self,
        query: str,
        topic_snapshots: Optional[List] = None,  # List[TopicSnapshot]
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """
        TheEye 凝视：意图识别、查询重写、话题路由

        Args:
            query: 用户查询字符串
            topic_snapshots: 活跃话题快照列表（用于路由和指代消解）
            identity: 用户身份标识

        Returns:
            EyeGazeResult: Eye 分析结果
        """
        if identity is None:
            identity = Identity()

        start_time = time.time()

        try:
            # 将 topic_snapshots 转换为文本格式
            from hivememory.engines.perception.context_converter import PerceptionContextConverter

            active_topics_menu = None
            if topic_snapshots:
                active_topics_menu = PerceptionContextConverter.snapshots_to_context_text(
                    topic_snapshots
                )

            result = await asyncio.to_thread(
                self._engine.process,
                query,
                None,  # context 参数已废弃，传 None
                active_topics_menu,
            )
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"TheEye 处理完成: "
                f"intent={result.intent.value}, "
                f"target_topic={result.target_topic}, "
                f"worth_saving={result.worth_saving}, "
                f"latency={result.processing_time_ms:.1f}ms"
            )

            return EyeGazeResult(
                intent=result.intent,
                rewritten_query=result.rewritten_query,
                search_keywords=result.search_keywords,
                worth_saving=result.worth_saving,
                raw_query=query,
                identity=identity,
                processing_time_ms=result.processing_time_ms,
                is_fallback=False,
                target_topic=result.target_topic,
            )
        except Exception as e:
            logger.error(f"TheEye 处理失败: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000
            return EyeGazeResult(
                intent=GatewayIntent.RAG,
                rewritten_query=query,
                search_keywords=[],
                worth_saving=False,
                raw_query=query,
                identity=identity,
                processing_time_ms=processing_time_ms,
                is_fallback=True,
                target_topic="NEW_TOPIC",
            )

    # ========== 被动观测模式 (Passive Observer Mode) ==========

    @property
    def observer_buffers(self) -> ObserverBufferManager:
        """访问 Observer Buffer 池"""
        return self._observer_buffers

    def ingest_user(
        self,
        content: str,
        identity: Identity,
        context: Optional[List[StreamMessage]] = None,  # Deprecated, kept for compatibility
    ) -> tuple[EyeGazeResult, Optional[InteractionPayload]]:
        """
        被动模式: 接收 user 消息

        Eye 分析 + 缓冲 (自动 flush 上一轮)。

        Args:
            content: 用户消息内容
            identity: 身份标识
            context: 对话历史上下文（已废弃，保留用于兼容）

        Returns:
            (gaze_result, flushed_payload):
                gaze_result: Eye 分析结果
                flushed_payload: 若触发 "Next User Turn" flush，返回上一轮 payload
        """
        gaze_result = asyncio.run(
            self.gaze(query=content, topic_snapshots=None, identity=identity)
        )
        buffer = self._observer_buffers.get_buffer(identity)
        flushed_payload = buffer.accept_user(content=content, gaze_result=gaze_result)
        return gaze_result, flushed_payload

    async def ingest_user_async(
        self,
        content: str,
        identity: Identity,
        context: Optional[List[StreamMessage]] = None,  # Deprecated, kept for compatibility
    ) -> tuple[EyeGazeResult, Optional[InteractionPayload]]:
        """
        被动模式: 接收 user 消息（异步版本）

        Args:
            content: 用户消息内容
            identity: 身份标识
            context: 对话历史上下文（已废弃，保留用于兼容）

        Returns:
            (gaze_result, flushed_payload):
                gaze_result: Eye 分析结果
                flushed_payload: 若触发 "Next User Turn" flush，返回上一轮 payload
        """
        gaze_result = await self.gaze(
            query=content, topic_snapshots=None, identity=identity
        )
        buffer = self._observer_buffers.get_buffer(identity)
        flushed_payload = buffer.accept_user(
            content=content, gaze_result=gaze_result
        )
        return gaze_result, flushed_payload

    def ingest_assistant(
        self,
        content: str,
        identity: Identity,
    ) -> None:
        """
        被动模式: 接收 assistant 消息，缓冲等待 flush

        Args:
            content: 助手消息内容
            identity: 身份标识
        """
        buffer = self._observer_buffers.get_buffer(identity)
        buffer.accept_assistant(content)

    def flush_session(self, identity: Identity) -> Optional[InteractionPayload]:
        """
        显式 flush 指定 session (Explicit EOF 触发器)

        Args:
            identity: 身份标识

        Returns:
            InteractionPayload 如果有数据，否则 None
        """
        buffer = self._observer_buffers.get_buffer(identity)
        return buffer.flush()

    def flush_idle_sessions(self, timeout_seconds: Optional[float] = None) -> List[InteractionPayload]:
        """
        扫描并 flush 所有超时的 buffer (Idle Timeout 触发器)

        Args:
            timeout_seconds: 超时阈值，默认使用 idle monitor 配置

        Returns:
            被 flush 的 InteractionPayload 列表
        """
        timeout = timeout_seconds or self._observer_idle_timeout
        return self._observer_buffers.flush_idle_buffers(timeout)

    def start_observer_idle_monitor(
        self,
        timeout_seconds: float = 30.0,
        scan_interval_seconds: float = 10.0,
        on_flush_callback=None,
    ) -> None:
        """
        启动 Observer Buffer 空闲超时监控

        Args:
            timeout_seconds: 超时阈值（秒）
            scan_interval_seconds: 扫描间隔（秒）
            on_flush_callback: flush 时的回调 (接收 InteractionPayload)
        """
        from apscheduler.schedulers.background import BackgroundScheduler

        if self._observer_idle_scheduler is not None:
            logger.warning("Observer idle monitor 已在运行")
            return

        self._observer_idle_timeout = timeout_seconds
        self._on_flush_callback = on_flush_callback
        self._observer_idle_scheduler = BackgroundScheduler()
        self._observer_idle_scheduler.add_job(
            self._scan_observer_idle_buffers,
            "interval",
            seconds=scan_interval_seconds,
        )
        self._observer_idle_scheduler.start()
        logger.info(
            f"Observer idle monitor 已启动: "
            f"timeout={timeout_seconds}s, interval={scan_interval_seconds}s"
        )

    def stop_observer_idle_monitor(self) -> None:
        """停止 Observer Buffer 空闲超时监控"""
        if self._observer_idle_scheduler is not None:
            self._observer_idle_scheduler.shutdown(wait=False)
            self._observer_idle_scheduler = None
            logger.info("Observer idle monitor 已停止")

    def _scan_observer_idle_buffers(self) -> None:
        """扫描并 flush 所有超时的 observer buffer"""
        payloads = self._observer_buffers.flush_idle_buffers(self._observer_idle_timeout)
        if self._bus:
            for payload in payloads:
                self._bus.emit("observer.idle_flushed", payload=payload)
        else:
            callback = getattr(self, "_on_flush_callback", None)
            if callback:
                for payload in payloads:
                    callback(payload)


__all__ = [
    "TheEye",
]
