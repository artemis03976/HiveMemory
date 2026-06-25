"""
HiveMemory - 语义流感知层 / MMU (Semantic Flow Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的 MMU（内存管理单元），管理多话题的生命周期。
    负责话题路由(route)、换入(swap-in)、换出(swap-out)和 LRU 驱逐。

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 5.0.0
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from hivememory.core.models import ActionReducer, Identity, TurnRecord
from hivememory.engines.perception.relay_controller import BaseRelayController
from hivememory.engines.perception.trigger_manager import TriggerManager
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    LogicalBlock,
    TopicMaterializeTask,
)
from hivememory.patchouli.memory_library.buffer import BufferState
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.core.protocol.models import InteractionPayload
from hivememory.utils.token_estimator import estimate_tokens

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


class SemanticFlowPerceptionLayer(BasePerceptionLayer):
    """
    语义流感知层 / MMU (Phase 5.0)

    作为短期记忆的内存管理单元 (MMU)，管理多话题的并发生命周期。
    TriggerManager 只负责决策和原子操作，不主动触发生成链路
    settle payload 通过返回值传递给调用方（PerceptionFamiliar）统一提交。
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        relay_controller: BaseRelayController,
        short_term_store: Optional["ShortTermMemoryStore"] = None,
    ):
        """
        初始化语义流感知层 (MMU)

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            relay_controller: 接力控制器 / Page Folding 摘要生成器
            short_term_store: 短期记忆存储（由 MemoryLibrary 创建后注入）
        """
        super().__init__()

        self.config = config

        self._relay_controller = relay_controller

        # 短期记忆存储必须由 MemoryLibrary 创建后注入，不允许引擎自行实例化
        if short_term_store is None:
            raise ValueError("未注入 short_term_store")
        self._short_term_store = short_term_store

        # TriggerManager 负责话题结算调度
        self._trigger_manager = TriggerManager(
            store=self._short_term_store,
            relay_controller=self._relay_controller,
        )

        logger.info("SemanticFlowPerceptionLayer (MMU) 初始化完成")

    # ========== 话题路由与管理 ==========

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> str:
        """
        确保目标话题存在并返回真实 topic_id。

        在 LLM 生成之前调用，将话题生命周期写操作提前执行：
        - 已有话题: 刷新 last_accessed_at 置顶
        - 新话题: 分配 UUID，保存 title/summary，检查 LRU 驱逐
        话题池与上下文读模型由 RetrievalFamiliar 负责读取。

        Args:
            target_topic_id: "NEW_TOPIC" 或已有 topic_id
            new_topic_title: Gateway 生成的新话题标题
            new_topic_summary: Gateway 生成的新话题摘要
            identity: 用户身份

        Returns:
            str: 可用的真实 topic_id
        """
        if target_topic_id == "NEW_TOPIC":
            topic_id = await self.create_new_topic(
                identity=identity,
                title=new_topic_title,
                summary=new_topic_summary,
            )
        else:
            if not self._short_term_store.topic_exists(target_topic_id):
                logger.warning(f"话题 {target_topic_id} 不存在，回退到创建新话题")
                topic_id = await self.create_new_topic(
                    identity=identity,
                    title=new_topic_title,
                    summary=new_topic_summary,
                )
            else:
                # 已有话题：刷新访问时间（置顶）
                topic_id = target_topic_id

        return topic_id

    async def create_new_topic(
        self,
        identity: Identity,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> str:
        """
        创建新话题。调用方负责在必要时提前执行 LRU 驱逐。
        """
        buffer = self._short_term_store.create_buffer(
            user_id=identity.user_id,
            topic_title=title or "新建话题",
            topic_summary=summary or "",
        )
        return buffer.topic_id

    # ========== 短期记忆上下文摄入 ==========

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> Tuple[str, Optional[TopicMaterializeTask]]:
        """
        MMU 核心方法：路由到指定话题并摄入载荷。

        Returns:
            (real_topic_id, TopicMaterializeTask | None)
            调用方负责将 TopicMaterializeTask 提交给生成链路。
        """
        # 重新检查创建情况，避免预创建后某些错误导致的异常
        topic_id = await self.prepare_topic(
            target_topic_id=topic_id,
            new_topic_title=None,
            new_topic_summary=None,
            identity=payload.identity,
        )
        settle_payload = await self.ingest_payload(payload, topic_id)
        self._short_term_store.set_last_active_topic(topic_id)
        return topic_id, settle_payload

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
    ) -> Optional[TopicMaterializeTask]:
        """
        摄入完整交互载荷。

        Returns:
            如发生 TOKEN_OVERFLOW 结算则返回 TopicMaterializeTask，否则 None
        """
        if not payload.turn_events:
            raise ValueError(
                "InteractionPayload.turn_events is required; "
                "legacy assistant_message fallback has been removed."
            )

        clean_text = payload.assistant_final_text or ""
        actions = ActionReducer.reduce(payload.turn_events)
        traces = payload.mtp_traces

        # 2. 构建 LogicalBlock
        block = LogicalBlock(
            turn=TurnRecord(
                identity=payload.identity,
                user_query=payload.user_message,
                rewritten_query=payload.rewritten_query,
                assistant_final_text=payload.assistant_final_text or clean_text,
                turn_events=payload.turn_events,
                actions=actions,
                semantic_traces=traces,
            ),
            worth_saving=payload.worth_saving,
        )

        # 2.5 计算 block 的 total_tokens
        block.total_tokens = (
            estimate_tokens(block.user_query)
            + estimate_tokens(block.assistant_final_text)
            + sum(
                estimate_tokens(t.query or "") + estimate_tokens(t.target or "")
                for t in block.semantic_traces
            )
        )

        # 3. 添加 block（被动流；主动生成由 finalize 直驱，不经此路径）
        self._short_term_store.add_block(topic_id, block)
        
        # 4. Page Folding 检查（token 溢出时压缩旧 blocks）
        settle_payload = await self._maybe_fold_pages(topic_id)
        
        # 重置状态
        self._short_term_store.update_metadata(topic_id, state=BufferState.IDLE)

        return settle_payload

    # ========== 上下文溢出检查 ==========

    async def _maybe_fold_pages(self, topic_id: str) -> Optional[TopicMaterializeTask]:
        """
        Page Folding: token 溢出时触发 Compact 操作
        """
        topic_data = self._short_term_store.get_topic_data(topic_id, touch=False)
        if topic_data is None:
            return None

        threshold = self.config.fold_token_threshold

        logger.debug(
            f"_maybe_fold_pages: topic_id={topic_id}, "
            f"total_tokens={topic_data.total_tokens}, threshold={threshold}, "
            f"blocks_count={topic_data.block_count}"
        )

        if topic_data.total_tokens <= threshold:
            return None

        logger.info(
            f"Token 溢出: topic_id={topic_id}, "
            f"total_tokens={topic_data.total_tokens} > threshold={threshold}"
        )
        return await self._trigger_manager.resolve_topic(
            FlushEvent(topic_id=topic_id, reason=FlushReason.TOKEN_OVERFLOW)
        )

    # ========== 话题结算原语 ==========

    async def settle_topic(
        self,
        topic_id: str,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> Optional[TopicMaterializeTask]:
        """
        原子话题结算，不含策略判断。由 PerceptionFamiliar 调用。
        
        Returns:
            TopicMaterializeTask | None
        """
        return await self._trigger_manager.resolve_topic(
            FlushEvent(topic_id=topic_id, reason=reason)
        )

    def swap_out_topic(self, topic_id: str) -> bool:
        """
        显式换出指定话题，返回是否存在该话题。
        """
        return self._short_term_store.pop_buffer(topic_id) is not None

    def discard_if_empty(self, topic_id: str) -> bool:
        """话题存在且无 blocks 时驱逐并返回 True，否则返回 False。"""
        info = self._short_term_store.get_buffer_info(topic_id)
        if info.get("exists") and info.get("block_count", 0) == 0:
            self._short_term_store.pop_buffer(topic_id)
            logger.info(f"已清理无内容话题: {topic_id}")
            return True
        return False


class NullPerceptionLayer(BasePerceptionLayer):
    """Disabled perception layer with the same public surface as SemanticFlow."""

    async def ingest_payload(self, payload: InteractionPayload, topic_id: str) -> None:
        return None

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> Tuple[str, None]:
        return topic_id, None

    async def settle_topic(
        self,
        topic_id: str,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> Optional[TopicMaterializeTask]:
        return None

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> str:
        return target_topic_id

    def swap_out_topic(self, topic_id: str) -> None:
        return None


__all__ = [
    "SemanticFlowPerceptionLayer",
    "NullPerceptionLayer",
]
