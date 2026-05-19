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
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Tuple

from hivememory.core.models import Identity
from hivememory.engines.perception.models import FlushReason, ArchivePayload
from hivememory.engines.generation.models import GenerationRequest, GenerationContext
from hivememory.engines.generation.generation_transcript_builder import GenerationTranscriptBuilder
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.core.protocol.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine

logger = logging.getLogger(__name__)


class LibrarianCore:
    """
    帕秋莉·馆长本体 (Librarian Core)

    这是帕秋莉的本体，坐在书桌前，一边喝红茶一边处理堆积如山的借阅记录。

    遵循显式依赖注入原则：所有子组件必须通过构造函数传入，
    不在内部实例化依赖项。由 PatchouliRuntime 负责组装和注入。

    职责:
        1. 接收 Eye 传来的感知信号 (Anchors)
        2. 维护 Buffer 和漂移检测
        3. 调用 Generation 引擎写书
        4. 调用 Lifecycle 引擎修书

    使用示例:
        >>> # 推荐：通过 PatchouliRuntime 使用
        >>> from hivememory.patchouli import PatchouliRuntime
        >>> runtime = PatchouliRuntime()
        >>> core = runtime.librarian_core
        >>>
        >>> # 高级：手动注入组件
        >>> core = LibrarianCore(
        ...     storage=storage,
        ...     lifecycle_engine=lifecycle_engine,
        ... )
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        bus: Optional[Any] = None,
        lifecycle_engine: Optional["MemoryLifecycleEngine"] = None,
        perception_layer: Optional["BasePerceptionLayer"] = None,
        generation_engine: Optional["MemoryGenerationEngine"] = None,
    ):
        """
        初始化馆长本体

        Args:
            storage: Qdrant 存储实例
            bus: 兼容保留参数（当前未使用）
            lifecycle_engine: 记忆生命周期引擎（预构建，由 PatchouliRuntime 注入）
            perception_layer: 感知层实例（用于直接调用）
            generation_engine: 生成引擎实例（用于直接调用）

        """
        self.storage = storage
        self._bus = bus
        self.lifecycle_engine = lifecycle_engine
        self.perception_layer = perception_layer
        self.generation_engine = generation_engine

        if self.perception_layer and hasattr(self.perception_layer, "set_generation_callback"):
            self.perception_layer.set_generation_callback(self._on_generate_memory)

        logger.info("LibrarianCore 初始化完成")

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        target_topic_id: str = "NEW_TOPIC",
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
            target_topic_id: 路由目标话题 ID 或 "NEW_TOPIC" (由 TheEye 决定)
        """
        logger.info(
            f"LibrarianCore 摄入交互载荷: "
            f"user='{payload.user_message[:30]}...', "
            f"target_topic_id={target_topic_id}, "
            f"traces={len(payload.mtp_traces)}, "
            f"write_focus={'YES' if payload.write_focus else 'NO'}, "
            f"update_focus={'YES' if payload.update_focus else 'NO'}"
        )

        # 直接调用感知层，感知层内部自动检测触发条件并调用回调
        if self.perception_layer:
            await self.perception_layer.route_and_ingest(target_topic_id, payload)
        else:
            logger.warning("perception_layer 未注入，跳过感知处理")

    async def ingest_interaction(
        self,
        payload: InteractionPayload,
        target_topic_id: str = "NEW_TOPIC",
    ) -> None:
        """兼容别名：旧代码仍可通过 ingest_interaction 调用。"""
        await self.submit_interaction(payload, target_topic_id=target_topic_id)

    async def _on_generate_memory(self, payload: "ArchivePayload") -> None:
        """
        感知层 Archive 回调（TriggerManager 触发）

        接收 TriggerManager 通过 asyncio.create_task() 调用。
        Phase 3: 使用 GenerationContext（结构化生成视图）作为 generation 主路径。

        根据 focus 信号选择 GenerationEngine 的处理模式:
            - Mode A (默认): 无 focus，普通记忆提取
            - Mode B (WRITE): 携带 write_focus，定向记忆生成
            - Mode C (UPDATE): 携带 update_focus，定向记忆更新

        每个 LogicalBlock 自行携带 identity，无需在此层面统一构建。

        Args:
            payload: ArchivePayload 对象，包含:
                - blocks: List[LogicalBlock] - 从 buffer flush 出的 blocks（每个携带 identity）
                - state_summary: str - 话题状态摘要
                - focus: write_focus 或 update_focus (仅 MTP_WRITE/UPDATE 时有值)
                - reason: FlushReason - flush 触发原因
                - topic_id: str - 话题 ID
                - user_id: Optional[str] - 用户 ID
        """
        try:
            focus = payload.focus
            reason = payload.reason

            # 提取 write_focus / update_focus
            write_focus = None
            update_focus = None
            if reason == FlushReason.MTP_WRITE:
                write_focus = focus
            elif reason == FlushReason.MTP_UPDATE:
                update_focus = focus

            # Phase 3: 构建结构化生成上下文（主路径）
            gen_context = self._build_generation_context(payload.blocks, payload.state_summary)

            # 只有纯 Mode A 且无上下文时才跳过；Mode B/C 允许空背景走 focus fallback
            if not gen_context.turns and write_focus is None and update_focus is None:
                logger.warning("空对话轮次，跳过处理")
                return

            # 根据 focus 信号构建对应模式的 GenerationRequest
            if reason == FlushReason.MTP_WRITE and write_focus is not None:
                logger.info(
                    f"LibrarianCore 处理 MTP_WRITE flush: "
                    f"{len(gen_context.turns)} 轮对话上下文"
                )
                request = GenerationRequest(
                    context=gen_context,
                    write_focus=write_focus,
                )
            elif reason == FlushReason.MTP_UPDATE and update_focus is not None:
                logger.info(
                    f"LibrarianCore 处理 MTP_UPDATE flush: "
                    f"alias='{update_focus.target_alias}', "
                    f"{len(gen_context.turns)} 轮对话上下文"
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
                    context=gen_context,
                    update_focus=update_focus,
                )
            else:
                # Mode A: 普通记忆提取
                logger.info(
                    f"LibrarianCore 开始处理 {len(gen_context.turns)} 轮对话..."
                )
                request = GenerationRequest(context=gen_context)

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

    def _build_generation_context(
        self,
        blocks: List[Any],
        state_summary: str = "",
    ) -> "GenerationContext":
        """
        Phase 3 主路径: 构建结构化记忆生成上下文。

        Args:
            blocks: LogicalBlock 列表
            state_summary: 话题状态摘要

        Returns:
            GenerationContext: 结构化生成视图
        """
        builder = GenerationTranscriptBuilder()
        return builder.build_context(blocks, state_summary=state_summary)

    # ========== 生命周期管理 API (未来扩展) ==========

    def start_gardening(self):
        """
        开启定时维护模式

        未来实现：调用 MemoryLifecycleEngine 进行定期维护
        """
        # TODO: 实现定时维护模式
        logger.warning("定时维护模式尚未实现")

    # ========== 感知层代理 API ==========

    def get_active_topics_snapshots(
        self,
        identity: Identity,
    ) -> List:  # List[TopicSnapshot]
        """
        获取活跃话题快照列表（代理感知层接口）

        Args:
            identity: 用户身份标识

        Returns:
            List[TopicSnapshot]: 话题快照列表
        """
        if self.perception_layer:
            return self.perception_layer.get_active_topics_snapshots(identity)
        logger.warning("perception_layer 未注入，返回空快照列表")
        return []

    async def manual_archive_topic(
        self,
        topic_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        手动触发话题结算 (Archive + Compact)

        代理感知层的 manual_trigger 接口，供上层统一调用。

        Args:
            topic_id: 目标话题 ID。如果为 None，使用 last_active_topic_id。

        Returns:
            Dict: 包含 success, topic_id, message, blocks_archived 的结果字典
        """
        if self.perception_layer:
            return await self.perception_layer.manual_trigger(topic_id)

        logger.warning("perception_layer 未注入，manual_archive_topic 失败")
        return {
            "success": False,
            "topic_id": topic_id or "unknown",
            "message": "perception_layer 未注入",
            "blocks_archived": 0,
        }

    async def manual_trigger(
        self,
        topic_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """兼容别名：旧代码仍可通过 manual_trigger 调用。"""
        return await self.manual_archive_topic(topic_id)

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        预创建/刷新话题（代理感知层接口），同时获取话题上下文

        Args:
            target_topic_id: "NEW_TOPIC" 或已有 topic_id
            new_topic_title: Gateway 生成的新话题标题
            new_topic_summary: Gateway 生成的新话题摘要
            identity: 用户身份

        Returns:
            (real_topic_id, pool_snapshot, topic_context)
        """
        if self.perception_layer:
            return await self.perception_layer.prepare_topic(
                target_topic_id, new_topic_title, new_topic_summary, identity
            )
        logger.warning("perception_layer 未注入，prepare_topic 失败")
        return target_topic_id, {"topics": [], "max_resident_topics": 5, "current_count": 0}, {"state_summary": "", "blocks": [], "total_tokens": 0, "title": ""}


__all__ = [
    "LibrarianCore",
]
