"""
HiveMemory - 语义流感知层 / MMU (Semantic Flow Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的无状态摄入编排层：把 ``InteractionPayload`` 构造为不可变
    ``LogicalBlock``，通过 Interaction apply journal 提供 retry 幂等，并在
    token 溢出时驱动 TopicBufferService 执行 Page Folding compact。

    Topic 状态、活跃池与 settle/evict 生命周期由 ``TopicBufferService``
    （Patchouli 领域服务）唯一拥有；本层不持有 Store、领域锁或队列。

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 6.0.0
"""

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from hivememory.core.models import (
    ActionReducer,
    IdentityScope,
    LogicalBlock,
    TurnRecord,
    WorkspaceAssetRef,
    require_identity_scope,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
)
from hivememory.patchouli.errors import TopicBusyError
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.utils.token_estimator import estimate_tokens

if TYPE_CHECKING:
    from hivememory.patchouli.services.topic_buffer import TopicBufferService

logger = logging.getLogger(__name__)


def _compute_apply_digest(
    block: LogicalBlock,
    asset_id_and_refs,
    model_used: str | None,
    identity_scope: IdentityScope,
) -> str:
    """计算一次 Interaction apply 的稳定输入摘要。

    digest 只覆盖可稳定重建的 canonical 输入（block 事实、binding refs 与参与
    原子 apply 的 metadata），刻意排除 ``block_id``/``created_at``/``bound_at``
    等 retry 时重新生成的随机或时钟值。它只用于判断同一 ``interaction_id`` 的
    retry 是否等价，不可作为 used refs 或 settlement 查询依据。
    """
    turn_dump = block.turn.model_dump(mode="json")
    # turn_id 与 block_id/created_at 一样，是 retry 时重新生成的随机标识，
    # 不参与等价性判断。
    turn_dump.pop("turn_id", None)
    canonical = {
        # Workspace 是 Store apply 的寻址边界；只依赖 block 内的 actor identity
        # 会把同一 interaction 在不同 Workspace 的提交误判为等价 retry。
        "identity_scope": identity_scope.model_dump(mode="json"),
        "turn": turn_dump,
        "total_tokens": block.total_tokens,
        "worth_saving": block.worth_saving,
        "gateway_intent": block.gateway_intent,
        "model_used": model_used or "",
        "asset_refs": sorted(
            (asset_id, asset_ref.token)
            for asset_id, asset_ref in asset_id_and_refs
        ),
    }
    payload = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SemanticFlowPerceptionLayer(BasePerceptionLayer):
    """
    语义流感知层 / MMU

    无状态的短期记忆摄入编排层。Topic 状态与活跃池由 ``TopicBufferService``
    唯一拥有；settle 的 generation admission 由调用方（PerceptionFamiliar）
    在领域锁外提交。
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        relay_controller,
        topic_buffer: "TopicBufferService",
        interaction_journal: InMemoryInteractionApplyJournal,
    ):
        """
        初始化语义流感知层 (MMU)

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            relay_controller: 接力控制器 / Page Folding 摘要生成器（供
                TopicBufferService 在领域锁外生成 compact 摘要）
            topic_buffer: Topic Buffer 领域服务（Topic 状态与活跃池唯一所有者）
            interaction_journal: interaction apply 的进程内幂等 journal
        """
        super().__init__()

        self.config = config

        self._relay_controller = relay_controller
        self._topic_buffer = topic_buffer
        self._interaction_journal = interaction_journal

        logger.info("SemanticFlowPerceptionLayer 初始化完成")

    # ========== 话题路由与管理 ==========

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity_scope: IdentityScope,
    ) -> str:
        """
        确保目标话题存在并返回真实 topic_id。

        在 LLM 生成之前调用，将话题生命周期写操作提前执行：
        - 已有话题: 返回其全局 ID（访问顺序由上层显式 touch/active 更新维护）
        - 新话题: 分配 UUID，保存 title/summary
        话题池与上下文读模型由 RetrievalFamiliar 负责读取。

        Args:
            target_topic_id: "NEW_TOPIC" 或已有 topic_id
            new_topic_title: Gateway 生成的新话题标题
            new_topic_summary: Gateway 生成的新话题摘要
            identity_scope: 已冻结的 Workspace 访问上下文

        Returns:
            str: 可用的真实 topic_id
        """
        # Topic ID 是全局身份，未知目标不能被投影成当前 Workspace 的新话题；
        # 由领域服务在创建/校验时显式拒绝。
        return self._topic_buffer.ensure_topic(
            identity_scope,
            target_topic_id,
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )

    async def create_new_topic(
        self,
        identity_scope: IdentityScope,
        title: str | None = None,
        summary: str | None = None,
    ) -> str:
        """
        创建新话题。调用方负责在必要时提前执行 LRU 驱逐。
        """
        identity_scope = require_identity_scope(identity_scope)
        data = self._topic_buffer.create_topic(
            identity_scope,
            topic_title=title,
            topic_summary=summary,
        )
        return data.topic_id

    # ========== 短期记忆上下文摄入 ==========

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
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
                    identity_scope=identity_scope,
                    interaction_id=interaction_id,
                    asset_id_and_refs=asset_id_and_refs,
                )
                return apply_record.topic_id, settle_payload

        # 重新检查创建情况，避免预创建后某些错误导致的异常
        topic_id = await self.prepare_topic(
            target_topic_id=topic_id,
            new_topic_title=None,
            new_topic_summary=None,
            identity_scope=identity_scope,
        )
        settle_payload = await self.ingest_payload(
            payload,
            topic_id,
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            asset_id_and_refs=asset_id_and_refs,
        )
        return topic_id, settle_payload

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
        *,
        identity_scope: IdentityScope,
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
    ) -> TopicMaterializeTask | None:
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

        block = self._build_block(payload, identity_scope)
        digest = _compute_apply_digest(
            block,
            asset_id_and_refs,
            payload.model_used,
            identity_scope,
        )

        if interaction_id:
            apply_record = self._interaction_journal.get(interaction_id)
            if apply_record is not None:
                self._require_equivalent_retry(apply_record, topic_id, interaction_id, digest)
                if apply_record.stage is InteractionApplyStage.COMPLETED:
                    return None
                if apply_record.stage is InteractionApplyStage.LOCAL_COMPLETED:
                    return apply_record.settlement_to_submit
                if not self._topic_buffer.reserve_processing(identity_scope, topic_id):
                    raise TopicBusyError(
                        f"topic '{topic_id}' 正忙，无法继续 interaction 后置义务，可稍后重试"
                    )
                return await self._complete_interaction_post_apply(
                    payload,
                    topic_id,
                    identity_scope=identity_scope,
                    interaction_id=interaction_id,
                )

        if not self._topic_buffer.reserve_processing(identity_scope, topic_id):
            raise TopicBusyError(
                f"topic '{topic_id}' 正忙，无法原子摄入 interaction，可稍后重试"
            )

        try:
            self._topic_buffer.apply_interaction(
                identity_scope,
                topic_id,
                block,
                interaction_id=interaction_id,
                asset_id_and_refs=asset_id_and_refs,
                model_used=payload.model_used,
            )
        except Exception:
            self._topic_buffer.release_processing(identity_scope, topic_id)
            raise

        if interaction_id:
            # journal 必须紧跟实际写入点；后续 folding/总线异常发生时，retry 仍能去重。
            self._interaction_journal.record_interaction_applied(
                interaction_id,
                topic_id,
                digest,
            )

        return await self._complete_interaction_post_apply(
            payload,
            topic_id,
            identity_scope=identity_scope,
            interaction_id=interaction_id,
        )

    def _build_block(
        self,
        payload: InteractionPayload,
        identity_scope: IdentityScope,
    ) -> LogicalBlock:
        """从 payload 构造本轮不可变逻辑块（block_id/created_at 不在 digest 内）。"""
        clean_text = payload.assistant_final_text or ""
        actions = ActionReducer.reduce(payload.turn_events)
        traces = payload.mtp_traces

        turn = TurnRecord(
            identity=identity_scope.actor_identity,
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
        return LogicalBlock(
            turn=turn,
            total_tokens=total_tokens,
            worth_saving=payload.worth_saving,
        )

    @staticmethod
    def _require_equivalent_retry(
        apply_record,
        topic_id: str,
        interaction_id: str,
        digest: str,
    ) -> None:
        """校验 retry 与已记录 apply 等价：topic 一致且 input digest 一致。"""
        if apply_record.topic_id != topic_id:
            raise ValueError(
                f"interaction '{interaction_id}' was already applied to another topic"
            )
        if apply_record.input_digest != digest:
            raise ValueError(
                f"interaction '{interaction_id}' was already applied with different input"
            )

    async def _complete_interaction_post_apply(
        self,
        payload: InteractionPayload,
        topic_id: str,
        *,
        identity_scope: IdentityScope,
        interaction_id: str | None,
    ) -> TopicMaterializeTask | None:
        """完成 block 写入后的本地义务，并为外层 settlement admission 留存结果。"""
        try:
            # Page Folding 检查（token 溢出时压缩旧 blocks，复用当前 PROCESSING 预约）
            settle_payload = await self._maybe_fold_pages(identity_scope, topic_id)
        finally:
            # 摘要生成或 compact 应用失败也必须结束本次预约，避免 Topic 永久 busy。
            self._topic_buffer.release_processing(identity_scope, topic_id)

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
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicMaterializeTask | None:
        """
        Page Folding: token 溢出时触发 Compact 操作
        """
        topic_data = self._topic_buffer.get_topic(identity_scope, topic_id, touch=False)
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
        # 统一计划执行入口：TOKEN_OVERFLOW 在矩阵中只表达 compact，不 settle。
        execution = self._topic_buffer.handle_trigger(
            identity_scope,
            topic_id,
            TriggerReason.TOKEN_OVERFLOW,
            retain_recent_blocks=self.config.fold_retain_recent_blocks,
        )
        if execution.settlement is None:
            return None
        return execution.settlement.task


class NullPerceptionLayer(BasePerceptionLayer):
    """Disabled perception layer with the same public surface as SemanticFlow."""

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
        *,
        identity_scope: IdentityScope,
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
    ) -> None:
        return None

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
    ) -> tuple[str, None]:
        return topic_id, None

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity_scope: IdentityScope,
    ) -> str:
        return target_topic_id


__all__ = [
    "SemanticFlowPerceptionLayer",
    "NullPerceptionLayer",
]
