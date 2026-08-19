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
from typing import TYPE_CHECKING

from hivememory.core.models import (
    ActionReducer,
    BufferState,
    LogicalBlock,
    TurnRecord,
    WorkspaceAccessContext,
    WorkspaceTopicKey,
    require_workspace_access_context,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    TopicMaterializeTask,
)
from hivememory.engines.perception.relay_controller import BaseRelayController
from hivememory.engines.perception.trigger_manager import TriggerManager
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
)
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.utils.token_estimator import estimate_tokens

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
        short_term_store: "ShortTermMemoryStore",
        interaction_journal: InMemoryInteractionApplyJournal,
    ):
        """
        初始化语义流感知层 (MMU)

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            relay_controller: 接力控制器 / Page Folding 摘要生成器
            short_term_store: 短期记忆存储（由 MemoryLibrary 创建后注入）
            interaction_journal: interaction apply 的进程内幂等 journal
        """
        super().__init__()

        self.config = config

        self._relay_controller = relay_controller

        self._short_term_store = short_term_store
        self._interaction_journal = interaction_journal

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
        new_topic_title: str | None,
        new_topic_summary: str | None,
        access_context: WorkspaceAccessContext,
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
            access_context: 已冻结的 Workspace 访问上下文

        Returns:
            str: 可用的真实 topic_id
        """
        if target_topic_id == "NEW_TOPIC":
            topic_id = await self.create_new_topic(
                access_context=access_context,
                title=new_topic_title,
                summary=new_topic_summary,
            )
        else:
            if not self._short_term_store.topic_exists(access_context, target_topic_id):
                logger.warning(f"话题 {target_topic_id} 不存在，回退到创建新话题")
                topic_id = await self.create_new_topic(
                    access_context=access_context,
                    title=new_topic_title,
                    summary=new_topic_summary,
                )
            else:
                # 已有话题：刷新访问时间（置顶）
                topic_id = target_topic_id

        return topic_id

    async def create_new_topic(
        self,
        access_context: WorkspaceAccessContext,
        title: str | None = None,
        summary: str | None = None,
    ) -> str:
        """
        创建新话题。调用方负责在必要时提前执行 LRU 驱逐。
        """
        access_context = require_workspace_access_context(access_context)
        buffer = self._short_term_store.create_buffer(
            access_context,
            topic_title=title or "新建话题",
            topic_summary=summary or "",
        )
        return buffer.topic_id

    # ========== 短期记忆上下文摄入 ==========

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
        interaction_id: str | None = None,
    ) -> tuple[str, TopicMaterializeTask | None]:
        """
        MMU 核心方法：路由到指定话题并摄入载荷。

        Returns:
            (real_topic_id, TopicMaterializeTask | None)
            调用方负责将 TopicMaterializeTask 提交给生成链路。
        """
        # consumer 侧先查 apply journal。已写入但尚未完成 settlement admission 的
        # interaction 继续执行后置义务，而不是把「block 已写入」误当成全部完成。
        if interaction_id:
            apply_record = self._interaction_journal.get(interaction_id)
            if apply_record is not None:
                settle_payload = await self.ingest_payload(
                    payload,
                    apply_record.topic_id,
                    interaction_id=interaction_id,
                )
                self._short_term_store.set_last_active_topic(
                    payload.access_context,
                    apply_record.topic_id,
                )
                return apply_record.topic_id, settle_payload

        # 重新检查创建情况，避免预创建后某些错误导致的异常
        topic_id = await self.prepare_topic(
            target_topic_id=topic_id,
            new_topic_title=None,
            new_topic_summary=None,
            access_context=payload.access_context,
        )
        settle_payload = await self.ingest_payload(
            payload,
            topic_id,
            interaction_id=interaction_id,
        )
        self._short_term_store.set_last_active_topic(payload.access_context, topic_id)
        return topic_id, settle_payload

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
        interaction_id: str | None = None,
    ) -> TopicMaterializeTask | None:
        """
        摄入完整交互载荷。

        Returns:
            如发生 TOKEN_OVERFLOW 结算则返回 TopicMaterializeTask，否则 None
        """
        if interaction_id:
            apply_record = self._interaction_journal.get(interaction_id)
            if apply_record is not None:
                if apply_record.topic_id != topic_id:
                    raise ValueError(
                        f"interaction '{interaction_id}' was already applied to another topic"
                    )
                if apply_record.stage is InteractionApplyStage.COMPLETED:
                    return None
                if apply_record.stage is InteractionApplyStage.LOCAL_COMPLETED:
                    return apply_record.settlement_to_submit
                return await self._complete_interaction_post_apply(
                    payload,
                    topic_id,
                    interaction_id=interaction_id,
                )

        if not payload.turn_events:
            raise ValueError(
                "InteractionPayload.turn_events is required; "
                "legacy assistant_message fallback has been removed."
            )

        clean_text = payload.assistant_final_text or ""
        actions = ActionReducer.reduce(payload.turn_events)
        traces = payload.mtp_traces

        # 2. 先构建只读 TurnRecord，再一次性构建 LogicalBlock
        turn = TurnRecord(
            identity=payload.identity,
            user_query=payload.user_message,
            rewritten_query=payload.rewritten_query,
            assistant_final_text=payload.assistant_final_text or clean_text,
            turn_events=payload.turn_events,
            actions=actions,
            semantic_traces=traces,
        )
        total_tokens = (
            estimate_tokens(turn.user_query)
            + estimate_tokens(turn.assistant_final_text)
            + sum(
                estimate_tokens(trace.query or "")
                + estimate_tokens(trace.target or "")
                for trace in turn.semantic_traces
            )
        )
        block = LogicalBlock(
            turn=turn,
            total_tokens=total_tokens,
            worth_saving=payload.worth_saving,
        )

        # 3. 添加 block（被动流；主动生成由 finalize 直驱，不经此路径）
        topic_key = WorkspaceTopicKey.from_access_context(payload.access_context, topic_id)
        self._short_term_store.add_block(topic_key, block)
        if interaction_id:
            # journal 必须紧跟实际写入点；后续 folding/总线异常发生时，retry 仍能去重。
            self._interaction_journal.record_block_applied(interaction_id, topic_id)

        return await self._complete_interaction_post_apply(
            payload,
            topic_id,
            interaction_id=interaction_id,
        )

    async def _complete_interaction_post_apply(
        self,
        payload: InteractionPayload,
        topic_id: str,
        *,
        interaction_id: str | None,
    ) -> TopicMaterializeTask | None:
        """完成 block 写入后的本地义务，并为外层 settlement admission 留存结果。"""
        # 3.1 若 payload 携带了 model_used（来自 ModelRegistry 解析结果），更新到 buffer
        if payload.model_used:
            self._short_term_store.update_model_used(
                WorkspaceTopicKey.from_access_context(payload.access_context, topic_id),
                payload.model_used,
            )

        # 4. Page Folding 检查（token 溢出时压缩旧 blocks）
        topic_key = WorkspaceTopicKey.from_access_context(payload.access_context, topic_id)
        settle_payload = await self._maybe_fold_pages(topic_key)

        # 重置状态
        self._short_term_store.update_metadata(topic_key, state=BufferState.IDLE)

        if interaction_id:
            self._interaction_journal.record_local_completed(
                interaction_id,
                topic_id,
                settle_payload,
            )

        return settle_payload

    # ========== 上下文溢出检查 ==========

    async def _maybe_fold_pages(
        self,
        topic_key: WorkspaceTopicKey,
    ) -> TopicMaterializeTask | None:
        """
        Page Folding: token 溢出时触发 Compact 操作
        """
        topic_data = self._short_term_store.get_topic_data_by_key(topic_key, touch=False)
        if topic_data is None:
            return None

        threshold = self.config.fold_token_threshold

        logger.debug(
            f"_maybe_fold_pages: topic_id={topic_key.topic_id}, "
            f"total_tokens={topic_data.total_tokens}, threshold={threshold}, "
            f"blocks_count={topic_data.block_count}"
        )

        if topic_data.total_tokens <= threshold:
            return None

        logger.info(
            f"Token 溢出: topic_id={topic_key.topic_id}, "
            f"total_tokens={topic_data.total_tokens} > threshold={threshold}"
        )
        return await self._trigger_manager.resolve_topic(
            FlushEvent(topic_key=topic_key, reason=FlushReason.TOKEN_OVERFLOW),
            retain_recent_blocks=self.config.fold_retain_recent_blocks,
        )

    # ========== 话题结算原语 ==========

    async def settle_topic(
        self,
        topic_key: WorkspaceTopicKey,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> TopicMaterializeTask | None:
        """
        原子话题结算，不含策略判断。由 PerceptionFamiliar 调用。

        Returns:
            TopicMaterializeTask | None
        """
        return await self._trigger_manager.resolve_topic(
            FlushEvent(topic_key=topic_key, reason=reason)
        )

    def swap_out_topic(self, topic_key: WorkspaceTopicKey) -> bool:
        """
        显式换出指定话题，返回是否存在该话题。
        """
        return self._short_term_store.pop_buffer_by_key(topic_key) is not None

    def discard_if_empty(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> bool:
        """话题存在且无 blocks 时驱逐并返回 True，否则返回 False。"""
        info = self._short_term_store.get_buffer_info(access_context, topic_id)
        if info.get("exists") and info.get("block_count", 0) == 0:
            self._short_term_store.pop_buffer(access_context, topic_id)
            logger.info(f"已清理无内容话题: {topic_id}")
            return True
        return False


class NullPerceptionLayer(BasePerceptionLayer):
    """Disabled perception layer with the same public surface as SemanticFlow."""

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
        interaction_id: str | None = None,
    ) -> None:
        return None

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
        interaction_id: str | None = None,
    ) -> tuple[str, None]:
        return topic_id, None

    async def settle_topic(
        self,
        topic_key: WorkspaceTopicKey,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> TopicMaterializeTask | None:
        return None

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        access_context: WorkspaceAccessContext,
    ) -> str:
        return target_topic_id

    def swap_out_topic(self, topic_key: WorkspaceTopicKey) -> None:
        return None


__all__ = [
    "SemanticFlowPerceptionLayer",
    "NullPerceptionLayer",
]
