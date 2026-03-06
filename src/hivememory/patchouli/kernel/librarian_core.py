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
import inspect
from typing import List, Optional, Callable, TYPE_CHECKING, Dict, Any

from hivememory.core.models import Identity, StreamMessage
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
        perception_layer: Optional["BasePerceptionLayer"] = None,
        generation_engine: Optional["MemoryGenerationEngine"] = None,
    ):
        """
        初始化馆长本体

        Args:
            storage: Qdrant 存储实例
            bus: SystemBus 实例（可选，用于外部通信）
            lifecycle_engine: 记忆生命周期引擎（预构建，由 PatchouliKernel 注入）
            perception_layer: 感知层实例（用于直接调用）
            generation_engine: 生成引擎实例（用于直接调用）

        """
        self.storage = storage
        self._bus = bus
        self.lifecycle_engine = lifecycle_engine
        self.perception_layer = perception_layer
        self.generation_engine = generation_engine

        # Flush 事件观察者列表
        self._flush_observers: List[FlushObserver] = []

        if self.perception_layer and hasattr(self.perception_layer, "set_generation_callback"):
            self.perception_layer.set_generation_callback(self._on_generate_memory)

        logger.info("LibrarianCore 初始化完成")

    def add_flush_observer(self, observer: FlushObserver) -> None:
        """添加 Flush 事件观察者"""
        self._flush_observers.append(observer)

    def remove_flush_observer(self, observer: FlushObserver) -> None:
        """移除 Flush 事件观察者"""
        if observer in self._flush_observers:
            self._flush_observers.remove(observer)

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    async def ingest_interaction(
        self,
        payload: InteractionPayload,
        target_topic: str = "NEW_TOPIC",
    ) -> None:
        """
        Kernel 模式主入口: 摄入完整交互载荷

        通过感知层 MMU 的 route_and_ingest 进行话题路由后摄入。
        感知层内部构建 LogicalBlock，检查 URGENT 信号并推送至 buffer。
        buffer 检测到 URGENT 标记后立即触发 flush，由 _on_generate_memory
        回调统一构建 GenerationRequest 并发送给 GenerationEngine。

        并发范式:
            内部直接调用子模块，不经过总线。

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

        # 直接调用感知层，感知层内部自动检测触发条件并调用回调
        if self.perception_layer:
            await self.perception_layer.route_and_ingest(target_topic, payload)
        else:
            logger.warning("perception_layer 未注入，跳过感知处理")

    async def _on_generate_memory(self, payload: Dict[str, Any]) -> None:
        """
        感知层 Archive 回调（TriggerManager 触发）

        接收 TriggerManager 通过 asyncio.create_task() 调用。
        Payload 包含从 buffer flush 出的 blocks，转换为 StreamMessage 后
        根据 focus 信号选择 GenerationEngine 的处理模式:
            - Mode A (默认): 无 focus，普通记忆提取
            - Mode B (WRITE): 携带 write_focus，定向记忆生成
            - Mode C (UPDATE): 携带 update_focus，定向记忆更新

        Args:
            payload: 包含以下键的字典:
                - blocks: List[LogicalBlock] - 从 buffer flush 出的 blocks
                - state_summary: str - 话题状态摘要
                - focus: write_focus 或 update_focus (仅 MTP_WRITE/UPDATE 时有值)
                - reason: FlushReason - flush 触发原因
        """
        try:
            blocks = payload.get("blocks", [])
            state_summary = payload.get("state_summary", "")
            focus = payload.get("focus")
            reason: FlushReason = payload.get("reason", FlushReason.IDLE_TIMEOUT)
            topic_id = payload.get("topic_id")
            identity = payload.get("identity")
            if identity is None and topic_id and self.perception_layer:
                try:
                    buffer = self.perception_layer.get_buffer(topic_id)
                    if buffer:
                        identity = buffer.identity
                except Exception:
                    identity = None

            messages = self._blocks_to_messages(blocks, identity)

            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            # 提取 write_focus / update_focus
            write_focus = None
            update_focus = None
            if reason == FlushReason.MTP_WRITE:
                write_focus = focus
            elif reason == FlushReason.MTP_UPDATE:
                update_focus = focus

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
                existing_result = self.storage.get_memory(_UUID(update_focus.target_uuid))
                existing = (
                    await existing_result
                    if inspect.isawaitable(existing_result)
                    else existing_result
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

            # 直接调用生成引擎
            if self.generation_engine:
                process_result = self.generation_engine.process(request)
                memories = (
                    await process_result
                    if inspect.isawaitable(process_result)
                    else process_result
                )
            else:
                logger.warning("generation_engine 未注入，跳过记忆生成")
                return

            if memories:
                logger.info(f"成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

    def _blocks_to_messages(
        self,
        blocks: List[Any],
        identity: Optional[Identity],
    ) -> List[StreamMessage]:
        messages: List[StreamMessage] = []
        for block in blocks:
            messages.extend(block.to_stream_messages(identity))
        return messages

    # ========== 生命周期管理 API (未来扩展) ==========

    def start_gardening(self):
        """
        开启定时维护模式

        未来实现：调用 MemoryLifecycleEngine 进行定期维护
        """
        # TODO: 实现定时维护模式
        logger.warning("定时维护模式尚未实现")

    # ========== 感知层代理 API ==========

    def get_active_topics_menu(self) -> List[Dict[str, Any]]:
        """
        获取活跃话题菜单（代理感知层接口）

        Returns:
            List[Dict]: 活跃话题列表，每个话题包含 topic_id, title, summary 等信息
        """
        if self.perception_layer:
            return self.perception_layer.get_active_topics_menu()
        logger.warning("perception_layer 未注入，返回空话题菜单")
        return []


__all__ = [
    "LibrarianCore",
]
