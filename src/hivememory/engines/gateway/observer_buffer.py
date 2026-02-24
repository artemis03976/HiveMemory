"""
HiveMemory Observer Session Buffer — 被动观测模式的会话缓冲器

改造自 block_builder.py 的 LogicalBlockBuilder 状态机，
用于将外部系统的离散消息配对为完整的 InteractionPayload。

状态机:
    IDLE -> (user msg) -> AWAITING_RESPONSE -> (assistant msg) -> SEALED
    SEALED -> flush() -> IDLE

三种 Flush 触发器 (参考 Passive.md §3.2):
    1. Next User Turn: 新 user 消息到达时，自动 flush 上一轮
    2. Idle Timeout: 超时后由外部调度器触发 flush
    3. Explicit EOF: 外部系统主动调用 flush()

作者: HiveMemory Team
版本: 1.0.0
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

from hivememory.core.models import Identity
from hivememory.engines.perception.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.patchouli.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)


class ObserverBufferState(str, Enum):
    """Observer Buffer 状态枚举"""
    IDLE = "idle"                       # 空闲，等待 user 消息
    AWAITING_RESPONSE = "awaiting"      # 已收到 user，等待 assistant
    SEALED = "sealed"                   # user+assistant 配对完成


class ObserverSessionBuffer:
    """
    单个 session 的观察者缓冲器

    改造自 LogicalBlockBuilder 的状态机模式:
    - 接收 (role, content) 离散消息
    - 配对 user + assistant 为一轮完整交互
    - 缓存 EyeGazeResult 用于 enrichment
    - flush 时构建 InteractionPayload

    与 LogicalBlockBuilder 的区别:
    - 输入: plain (role, content) 而非 StreamMessage
    - 输出: InteractionPayload 而非 LogicalBlock
    - 额外缓存: EyeGazeResult (Eye 分析结果)
    """

    def __init__(self, identity: Identity) -> None:
        self._identity = identity
        self._reset()

    def _reset(self) -> None:
        """重置到 IDLE 状态 (对应 LogicalBlockBuilder._reset)"""
        self._state = ObserverBufferState.IDLE
        self._user_content: Optional[str] = None
        self._assistant_parts: List[str] = []
        self._gaze_result: Optional[EyeGazeResult] = None
        self._last_activity: float = datetime.now().timestamp()

    # ========== 状态属性 ==========

    @property
    def state(self) -> ObserverBufferState:
        return self._state

    @property
    def is_idle(self) -> bool:
        """对应 LogicalBlockBuilder.is_empty"""
        return self._state == ObserverBufferState.IDLE

    @property
    def is_awaiting(self) -> bool:
        """对应 LogicalBlockBuilder.is_started"""
        return self._state == ObserverBufferState.AWAITING_RESPONSE

    @property
    def is_sealed(self) -> bool:
        """对应 LogicalBlockBuilder.is_complete"""
        return self._state == ObserverBufferState.SEALED

    @property
    def has_pending_round(self) -> bool:
        """是否有未 flush 的数据 (AWAITING 或 SEALED)"""
        return self._state != ObserverBufferState.IDLE

    @property
    def last_activity_time(self) -> float:
        return self._last_activity

    # ========== 消息接收 ==========

    def accept_user(
        self,
        content: str,
        gaze_result: Optional[EyeGazeResult] = None,
    ) -> Optional[InteractionPayload]:
        """
        接收 user 消息

        状态转换:
        - IDLE -> AWAITING_RESPONSE (正常开始新轮)
        - AWAITING_RESPONSE -> flush 上一轮 -> AWAITING_RESPONSE
        - SEALED -> flush 上一轮 -> AWAITING_RESPONSE

        对应 Passive.md §3.2 "Next User Turn" 触发器:
        当收到同一 Session 的下一条 user 消息时，
        说明上一轮必然已结束，立即打包上一轮数据。

        Args:
            content: 用户消息内容
            gaze_result: TheEye 分析结果 (可选)

        Returns:
            如果触发了 "Next User Turn" flush，返回上一轮的 payload；
            否则返回 None
        """
        flushed_payload = None

        if self.has_pending_round:
            flushed_payload = self._build_payload()
            self._reset()

        self._user_content = content
        self._gaze_result = gaze_result
        self._state = ObserverBufferState.AWAITING_RESPONSE
        self._last_activity = datetime.now().timestamp()

        logger.debug(
            f"Observer buffer 接收 user 消息: "
            f"session={self._identity.buffer_key}, "
            f"flushed_previous={flushed_payload is not None}"
        )

        return flushed_payload

    def accept_assistant(self, content: str) -> None:
        """
        接收 assistant 消息

        状态转换:
        - AWAITING_RESPONSE -> SEALED (首条 assistant)
        - SEALED -> SEALED (追加 assistant，多段回复场景)
        - IDLE -> 忽略 (无配对的 user，记录 warning)

        Args:
            content: 助手消息内容
        """
        if self._state == ObserverBufferState.IDLE:
            logger.warning(
                f"Observer buffer 收到 assistant 消息但无配对的 user 消息，忽略: "
                f"session={self._identity.buffer_key}, "
                f"content='{content[:50]}...'"
            )
            return

        self._assistant_parts.append(content)
        self._state = ObserverBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

        logger.debug(
            f"Observer buffer 接收 assistant 消息: "
            f"session={self._identity.buffer_key}, "
            f"parts_count={len(self._assistant_parts)}"
        )

    # ========== Flush ==========

    def flush(self) -> Optional[InteractionPayload]:
        """
        显式 flush: 构建 payload 并重置

        用于 Idle Timeout 和 Explicit EOF 触发器。

        Returns:
            InteractionPayload 如果有数据，否则 None
        """
        if self._state == ObserverBufferState.IDLE:
            return None

        payload = self._build_payload()
        self._reset()

        logger.debug(
            f"Observer buffer flush: session={self._identity.buffer_key}"
        )

        return payload

    def _build_payload(self) -> InteractionPayload:
        """
        构建 InteractionPayload (对应 LogicalBlockBuilder.build)

        被动模式特征 (参考 Passive.md §3.3):
        - mtp_traces: 空列表 (被动模式无 MTP 协议指令)
        - write_focus / update_focus: None
        - rewritten_query / worth_saving: 从缓存的 EyeGazeResult 提取
        """
        assistant_text = (
            "\n".join(self._assistant_parts)
            if self._assistant_parts
            else ""
        )

        return InteractionPayload(
            user_message=self._user_content or "",
            assistant_message=assistant_text,
            mtp_traces=[],
            write_focus=None,
            update_focus=None,
            identity=self._identity,
            rewritten_query=(
                self._gaze_result.rewritten_query
                if self._gaze_result else None
            ),
            worth_saving=(
                self._gaze_result.worth_saving
                if self._gaze_result else None
            ),
        )


class ObserverBufferManager:
    """
    Observer Buffer 池管理器

    按 Identity.buffer_key 管理多个 ObserverSessionBuffer，
    线程安全 (复用 SemanticBufferManager 的 RLock 模式)。
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, ObserverSessionBuffer] = {}
        self._lock = threading.RLock()
        logger.info("ObserverBufferManager 初始化完成")

    def get_buffer(self, identity: Identity) -> ObserverSessionBuffer:
        """获取或创建指定 session 的 buffer"""
        key = identity.buffer_key
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = ObserverSessionBuffer(identity=identity)
                logger.debug(f"创建新 observer buffer: {key}")
            return self._buffers[key]

    def remove_buffer(self, identity: Identity) -> None:
        """移除指定 session 的 buffer"""
        key = identity.buffer_key
        with self._lock:
            self._buffers.pop(key, None)

    def list_active_buffers(self) -> Dict[str, ObserverSessionBuffer]:
        """返回所有 buffer 的快照"""
        with self._lock:
            return dict(self._buffers)

    def flush_idle_buffers(self, timeout_seconds: float) -> List[InteractionPayload]:
        """
        扫描并 flush 所有超时的 buffer

        用于 Idle Timeout 触发器 (Passive.md §3.2):
        收到 Assistant 消息后，若超过 T 秒无新消息，判定本轮结束。

        Args:
            timeout_seconds: 超时阈值（秒）

        Returns:
            被 flush 的 InteractionPayload 列表
        """
        now = datetime.now().timestamp()
        payloads: List[InteractionPayload] = []

        with self._lock:
            for key, buf in list(self._buffers.items()):
                if buf.has_pending_round:
                    idle_duration = now - buf.last_activity_time
                    if idle_duration > timeout_seconds:
                        payload = buf.flush()
                        if payload:
                            payloads.append(payload)
                            logger.info(
                                f"Observer idle timeout flush: "
                                f"session={key}, "
                                f"idle={idle_duration:.1f}s"
                            )

        return payloads


__all__ = [
    "ObserverBufferState",
    "ObserverSessionBuffer",
    "ObserverBufferManager",
]
