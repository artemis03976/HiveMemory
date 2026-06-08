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

import inspect
import logging
from time import monotonic
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from hivememory.core.models import Identity
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.generation.models import GenerationContext
from hivememory.engines.perception.models import ArchivePayload
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.patchouli.services.memory_generation_tasks import (
    MemoryGenerationTaskController,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from hivememory.system.runtime.control import (
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
)

if TYPE_CHECKING:
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
    from hivememory.engines.perception.interfaces import BasePerceptionLayer

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
        task_registry: Optional[MemoryGenerationTaskRegistry] = None,
    ):
        self.storage = storage
        self._bus = bus
        self.lifecycle_engine = lifecycle_engine
        self.perception_layer = perception_layer
        self.generation_engine = generation_engine
        self._memory_task_controller = MemoryGenerationTaskController(
            storage=storage,
            bus=bus,
            generation_engine=generation_engine,
            task_registry=task_registry,
        )

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
            f"tasks={len(payload.materialize_tasks)}"
        )

        # 直接调用感知层，感知层内部自动检测触发条件并调用回调
        if self.perception_layer:
            await self.perception_layer.route_and_ingest(target_topic_id, payload)
        else:
            logger.warning("perception_layer 未注入，跳过感知处理")

    async def _on_generate_memory(self, payload: ArchivePayload) -> Optional[MemoryGenerationTask]:
        """感知层 Archive 回调 — Mode A 被动记忆提取。"""
        gen_context = self._build_generation_context(payload.blocks, payload.state_summary)
        if not gen_context.turns:
            logger.warning("空对话轮次，跳过处理")
            return None

        return await self._memory_task_controller.run_archive_generation(
            topic_id=payload.topic_id,
            gen_context=gen_context,
        )

    async def run_active_generation(
        self,
        tasks: List[PendingAtomMaterializeTask],
        topic_id: str,
    ) -> List[MemoryGenerationTask]:
        """Run MTP WRITE/UPDATE memory generation tasks in the background."""
        if not tasks:
            return []

        topic_context: Dict[str, Any] = {"state_summary": "", "blocks": []}
        if self.perception_layer is not None:
            topic_context = self.perception_layer.get_topic_context(topic_id)

        gen_context = self._build_generation_context(
            topic_context.get("blocks", []),
            topic_context.get("state_summary", ""),
        )
        return await self._memory_task_controller.run_active_generation(
            tasks,
            topic_id,
            gen_context=gen_context,
        )

    def get_task(self, task_id: str) -> Optional[MemoryGenerationTask]:
        return self._memory_task_controller.get_task(task_id)

    def list_tasks(self) -> List[MemoryGenerationTask]:
        return self._memory_task_controller.list_tasks()

    def cancel_task(self, task_id: str) -> bool:
        return self._memory_task_controller.cancel_task(task_id)

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

    # ========== 生命周期管理 API ==========

    async def run_gardening_once(self) -> Dict[str, Any]:
        """Run one lifecycle garbage-collection pass for the global scheduler."""
        start = monotonic()
        result = {
            "success": False,
            "archived_count": 0,
            "duration_ms": 0.0,
            "error": None,
        }

        if self.lifecycle_engine is None:
            result["error"] = "lifecycle_engine is not available"
            result["duration_ms"] = (monotonic() - start) * 1000
            logger.warning("Lifecycle gardening skipped: lifecycle_engine is not available")
            return result

        try:
            archived = self.lifecycle_engine.run_garbage_collection(force=False)
            if inspect.isawaitable(archived):
                archived = await archived
            result["success"] = True
            result["archived_count"] = int(archived or 0)
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Lifecycle gardening failed: %s", exc, exc_info=True)
        finally:
            result["duration_ms"] = (monotonic() - start) * 1000

        return result

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
