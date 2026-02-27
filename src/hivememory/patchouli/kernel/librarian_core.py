"""
帕秋莉·馆长本体 (Librarian Core)

定位：思考者与管理者
职责：
    - 接收 Eye 传来的感知信号 (Anchors)
    - 维护 Buffer 和漂移检测
    - 调用 Generation 引擎写书
    - 调用 Lifecycle 引擎修书

基于原 PatchouliAgent (agents/patchouli.py) 改造，专注于 Cold Path 处理

作者: HiveMemory Team
版本: 2.1
"""

from __future__ import annotations

import logging
from typing import List, Optional, Callable, TYPE_CHECKING

from hivememory.core.models import StreamMessage
from hivememory.engines.perception.models import FlushEvent, FlushReason, InteractionPayload
from hivememory.engines.generation.models import GenerationRequest
from hivememory.infrastructure.storage import QdrantMemoryStore

FlushObserver = Callable[[FlushEvent], None]

if TYPE_CHECKING:
    from hivememory.infrastructure.system_bus import SystemBus
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine

logger = logging.getLogger(__name__)


class LibrarianCore:
    """
    帕秋莉·馆长本体 (Librarian Core)

    这是帕秋莉的本体，坐在书桌前，一边喝红茶一边处理堆积如山的借阅记录。

    遵循显式依赖注入原则：所有子组件必须通过构造函数传入，
    不在内部实例化依赖项。由 PatchouliKernel 负责组装和注入。

    职责:
        1. 接收 Eye 传来的感知信号 (Anchors)
        2. 维护 Buffer 和漂移检测
        3. 调用 Generation 引擎写书
        4. 调用 Lifecycle 引擎修书

    使用示例:
        >>> # 推荐：通过 PatchouliKernel 使用
        >>> from hivememory.patchouli import PatchouliKernel
        >>> kernel = PatchouliKernel()
        >>> core = kernel.librarian_core
        >>>
        >>> # 高级：手动注入组件
        >>> core = LibrarianCore(
        ...     storage=storage,
        ...     bus=bus,
        ...     lifecycle_engine=lifecycle_engine,
        ... )
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        bus: Optional["SystemBus"] = None,
        lifecycle_engine: Optional["MemoryLifecycleEngine"] = None,
    ):
        """
        初始化馆长本体

        Args:
            storage: Qdrant 存储实例
            bus: SystemBus 实例，用于跨服务通信（替代 perception_layer + generation_engine）
            lifecycle_engine: 记忆生命周期引擎（预构建，由 PatchouliKernel 注入）

        """
        self.storage = storage
        self._bus = bus
        self.lifecycle_engine = lifecycle_engine

        # Flush 事件观察者列表
        self._flush_observers: List[FlushObserver] = []

        # 通过 bus 订阅感知层 flush 事件（替代 set_flush_callback）
        if self._bus:
            self._bus.subscribe("perception.flushed", self._on_perception_flush)

        logger.info("LibrarianCore 初始化完成")

    def add_flush_observer(self, observer: FlushObserver) -> None:
        """添加 Flush 事件观察者"""
        self._flush_observers.append(observer)

    def remove_flush_observer(self, observer: FlushObserver) -> None:
        """移除 Flush 事件观察者"""
        if observer in self._flush_observers:
            self._flush_observers.remove(observer)

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    def ingest_interaction(
        self,
        payload: InteractionPayload,
        target_topic: str = "NEW_TOPIC",
    ) -> None:
        """
        Kernel 模式主入口: 摄入完整交互载荷

        通过感知层 MMU 的 route_and_ingest 进行话题路由后摄入。
        感知层内部构建 LogicalBlock，检查 URGENT 信号并推送至 buffer。
        buffer 检测到 URGENT 标记后立即触发 flush，由 _on_perception_flush
        回调统一构建 GenerationRequest 并发送给 GenerationEngine。

        Args:
            payload: Kernel → Perception 的原子传输包
            target_topic: 路由目标话题 ID 或 "NEW_TOPIC" (由 TheEye 决定)
        """
        logger.info(
            f"LibrarianCore 摄入交互载荷: "
            f"user='{payload.user_message[:30]}...', "
            f"target_topic={target_topic}, "
            f"traces={len(payload.mtp_traces)}, "
            f"write_focus={'YES' if payload.write_focus else 'NO'}, "
            f"update_focus={'YES' if payload.update_focus else 'NO'}"
        )

        # 通过 MMU 路由到目标话题后摄入
        self._bus.request("perception.route_and_ingest", target_topic, payload)

    def _on_perception_flush(
        self,
        messages: List[StreamMessage],
        reason: FlushReason,
        write_focus=None,
        update_focus=None,
    ) -> None:
        """
        感知层 Flush 回调（统一接口）

        所有 GenerationRequest 均从此回调构建，包括 WRITE/UPDATE 模式。
        感知层通过 flush 事件携带 focus 控制信号，本回调根据信号选择
        GenerationEngine 的处理模式:
            - Mode A (默认): 无 focus，普通记忆提取
            - Mode B (WRITE): 携带 write_focus，定向记忆生成
            - Mode C (UPDATE): 携带 update_focus，定向记忆更新

        Args:
            messages: StreamMessage 列表 (从 buffer flush 出的完整上下文)
            reason: Flush 原因
            write_focus: WRITE 指令控制信号 (仅 MTP_WRITE flush 时传入)
            update_focus: UPDATE 指令控制信号 (仅 MTP_UPDATE flush 时传入)
        """
        try:
            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            # 根据 focus 信号构建对应模式的 GenerationRequest
            if reason == FlushReason.MTP_WRITE and write_focus is not None:
                logger.info(
                    f"LibrarianCore 处理 MTP_WRITE flush: "
                    f"{len(messages)} 条上下文消息"
                )
                request = GenerationRequest(
                    context_messages=messages,
                    write_focus=write_focus,
                )
            elif reason == FlushReason.MTP_UPDATE and update_focus is not None:
                logger.info(
                    f"LibrarianCore 处理 MTP_UPDATE flush: "
                    f"alias='{update_focus.target_alias}', "
                    f"{len(messages)} 条上下文消息"
                )
                # 加载目标记忆
                from uuid import UUID as _UUID
                existing = self._bus.request(
                    "storage.get_memory", _UUID(update_focus.target_uuid)
                )
                if existing is None:
                    logger.error(
                        f"UPDATE 目标记忆不存在: {update_focus.target_uuid}"
                    )
                    return
                update_focus.existing_memory = existing
                request = GenerationRequest(
                    context_messages=messages,
                    update_focus=update_focus,
                )
            else:
                # Mode A: 普通记忆提取
                logger.info(
                    f"LibrarianCore 开始处理 {len(messages)} 条消息..."
                )
                request = GenerationRequest(context_messages=messages)

            memories = self._bus.request("generation.process", request)

            if memories:
                logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

    # ========== 生命周期管理 API (未来扩展) ==========

    def start_gardening(self):
        """
        开启定时维护模式

        未来实现：调用 MemoryLifecycleEngine 进行定期维护
        """
        # TODO: 实现定时维护模式
        logger.warning("定时维护模式尚未实现")


__all__ = [
    "LibrarianCore",
]
