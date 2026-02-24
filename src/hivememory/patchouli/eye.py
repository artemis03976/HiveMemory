"""
帕秋莉·真理之眼 (The Eye of Patchouli)

定位：守门人与感知者
职责：
    - 流量入口和意图判断 (Hot)
    - 调用 GatewayService 获取原始结果
    - 处理 fallback、日志、数据转换
    - 产出 RetrievalRequest 和 Observation 供下游使用
    - 被动观测模式下管理 ObserverBuffer 池 (Passive Observer Mode)

作者: HiveMemory Team
版本: 2.3
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

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
    ):
        """
        初始化真理之眼

        Args:
            engine: Gateway 引擎实例
        """
        self._engine = engine
        self._observer_buffers = ObserverBufferManager()
        self._observer_idle_scheduler = None
        self._observer_idle_timeout: float = 30.0

        logger.info(f"TheEye 真理之眼初始化完成")

    def gaze(
        self,
        query: str,
        context: Optional[List[StreamMessage]] = None,
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """
        审视用户查询（真理之眼的主要入口方法）

        Eye 只负责感知和信息重整，不构建下游协议消息。
        返回 EyeGazeResult 供 Kernel 进行数据格式转换。

        Args:
            query: 用户原始查询
            context: 对话上下文（可选），用于指代消解
            identity: 身份标识对象

        Returns:
            EyeGazeResult: Eye 的统一输出模型
        """
        if identity is None:
            identity = Identity()

        start_time = time.time()

        try:
            # 调用数据操作层
            result = self._engine.process(query, context)

            # 业务逻辑：添加元信息
            result.processing_time_ms = (time.time() - start_time) * 1000

            # 业务逻辑：日志记录
            logger.info(
                f"TheEye 处理完成: "
                f"intent={result.intent.value}, "
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
            )

        except Exception as e:
            logger.error(f"TheEye 处理失败: {e}", exc_info=True)
            # Fallback 处理
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
        context: Optional[List[StreamMessage]] = None,
    ) -> tuple[EyeGazeResult, Optional[InteractionPayload]]:
        """
        被动模式: 接收 user 消息

        Eye 分析 + 缓冲 (自动 flush 上一轮)。

        Args:
            content: 用户消息内容
            identity: 身份标识
            context: 对话历史上下文

        Returns:
            (gaze_result, flushed_payload):
                gaze_result: Eye 分析结果
                flushed_payload: 若触发 "Next User Turn" flush，返回上一轮 payload
        """
        gaze_result = self.gaze(query=content, context=context, identity=identity)
        buffer = self._observer_buffers.get_buffer(identity)
        flushed_payload = buffer.accept_user(content=content, gaze_result=gaze_result)
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
        callback = getattr(self, "_on_flush_callback", None)
        if callback:
            for payload in payloads:
                callback(payload)


__all__ = [
    "TheEye",
]
