from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal
from uuid import UUID

from hivememory.components.work_queue import (
    WorkQueueCapacityError,
    WorkQueueStoppedError,
    WorkState,
)
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    RetrievalMode,
)
from hivememory.core.protocol.models import (
    InteractionPayload,
    RetrievalResponse,
)
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    InteractionSubmissionReceipt,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.control.pending_atom_settler import PendingAtomSettler
from hivememory.patchouli.runtime.bus import PatchouliBus

logger = logging.getLogger(__name__)

ActiveFinalizationStage = Literal[
    "interaction_admission",
    "interaction_apply",
]


class ActiveInteractionFinalizationError(RuntimeError):
    """Active interaction 未能跨过 admission/apply 硬成功边界。"""

    def __init__(
        self,
        *,
        interaction_id: str,
        stage: ActiveFinalizationStage,
        reason: str,
        work_state: WorkState | None = None,
        error_class: str | None = None,
    ) -> None:
        self.interaction_id = interaction_id
        self.stage = stage
        self.reason = reason
        self.work_state = work_state
        self.error_class = error_class
        super().__init__(f"Active interaction finalization failed at {stage}: {reason}")


class PatchouliService:
    """Patchouli 对外能力门面，承载 Agent prepare/finalize 与交互接纳编排。

    prepare 只做 Topic 与检索：Profile 解析、附件租借与记忆/附件编译由
    chat 任务进程在 CPU 分配时完成，不再出现在 Patchouli 公开路由上。

    访问边界（A1 访问边界返工第 4.6 节）：本门面是授权点以下的资源
    owner，阶段路由只接收任务进程在阶段授权后组装的 ``IdentityScope``，
    不接收访问 context，也不做 operation 检查；资源归属仍由本层与领域
    实现独立校验。
    """

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        interaction_queue: InteractionSubmissionQueue,
        pending_atom_settler: PendingAtomSettler | None = None,
    ) -> None:
        if interaction_queue is None:
            raise TypeError("interaction_queue is required")
        self._local_bus = bus
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        self._interaction_queue = interaction_queue
        self._active_finalizations: dict[
            str,
            asyncio.Task[list[MemoryGenerationTask]],
        ] = {}
        self._detached_finalizations: set[str] = set()

    async def prepare_agent_run(
        self,
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        gateway_decision: GatewayDecision,
        enable_memory_retrieval: bool = True,
    ) -> PreparedAgentRun:
        """准备本轮的 Topic 与未编译检索结果（prepare 只做 Topic 与检索）。

        返回的 PreparedAgentRun 携带话题准备结果与检索到的原始记忆原子；
        Profile 解析、附件租借与编译由任务进程在 CPU 分配时完成，用户消息
        与 Gateway 决定也由进程持有。prepare 失败时只清理本轮可能预创建的
        空话题。``identity_scope`` 是任务进程完成 ``resource.search`` 阶段
        授权后组装的可信 scope。
        """
        identity_scope = require_identity_scope(identity_scope)
        belong_to = identity_scope.workspace_identity
        real_topic_id: str | None = None
        is_new = gateway_decision.target_topic_id == "NEW_TOPIC"

        try:
            real_topic_id = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_PREPARE,
                target_topic_id=gateway_decision.target_topic_id,
                new_topic_title=gateway_decision.new_topic_title,
                new_topic_summary=gateway_decision.new_topic_summary,
                belong_to=belong_to,
            )
            pool_topics = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                belong_to=belong_to,
                include_empty=True,
            )
            topic_context = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_GET,
                real_topic_id,
                belong_to=belong_to,
            )

            retrieval_result = await self.retrieve_for_decision(
                gateway_decision,
                belong_to=belong_to,
                from_actor=identity_scope.actor_identity,
                enable_retrieval=enable_memory_retrieval,
            )

            return PreparedAgentRun(
                belong_to=belong_to,
                interaction_id=interaction_id,
                topic_id=real_topic_id,
                is_new_topic=is_new,
                topic_context=topic_context,
                pool_topics=pool_topics,
                retrieval_result=retrieval_result,
                storage_available=await self._local_bus.request(
                    PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
                ),
            )
        except Exception:
            # prepare 失败：沿既有路径清理可能预创建的空话题。
            if is_new and real_topic_id:
                await self._cleanup_empty_topic_if_needed(belong_to, real_topic_id)
            raise

    async def finalize_agent_run(
        self,
        prepared_run: PreparedAgentRun,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
    ) -> list[MemoryGenerationTask]:
        """原样提交封口好的交互记录，并把 post-apply 工作交给 Patchouli 持有。

        ``payload`` 由提交方（任务进程）组装并封口，与被动链路一致；finalize
        原样提交，不改写其内容。物化任务按 ``payload.materialize_tasks`` 派发；附件租借由进程持有并随进程关闭
        释放，finalize 不负责释放。``identity_scope`` 来自调用前的
        ``interaction.submit`` 阶段授权；公共边界拆分后，后台 continuation
        只携带归属与本次提交的发起者。
        """
        scope = require_identity_scope(identity_scope)
        if prepared_run.belong_to != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"interaction_id": prepared_run.interaction_id})
        continuation = self._active_finalizations.get(prepared_run.interaction_id)
        if continuation is None:
            continuation = asyncio.create_task(
                self._continue_active_finalization(prepared_run, payload, scope.actor_identity),
                name=f"active_finalize_{prepared_run.interaction_id[:8]}",
            )
            self._active_finalizations[prepared_run.interaction_id] = continuation
            continuation.add_done_callback(
                lambda completed, interaction_id=prepared_run.interaction_id: (
                    self._active_finalization_done(interaction_id, completed)
                )
            )

        # 调用方取消只中断当前等待；continuation 继续完成已接管的业务义务。
        try:
            return await asyncio.shield(continuation)
        except asyncio.CancelledError:
            if not continuation.done():
                self._detached_finalizations.add(prepared_run.interaction_id)
            raise

    async def _continue_active_finalization(
        self,
        prepared_run: PreparedAgentRun,
        payload: InteractionPayload,
        from_actor: ActorIdentity,
    ) -> list[MemoryGenerationTask]:
        try:
            receipt = await self._admit_active_interaction(prepared_run, payload, from_actor)
            await self._wait_active_interaction(prepared_run, receipt)
        except ActiveInteractionFinalizationError as error:
            if prepared_run.is_new_topic and (
                error.stage == "interaction_apply"
                or prepared_run.interaction_id in self._detached_finalizations
            ):
                await self._cleanup_empty_topic_if_needed(
                    prepared_run.belong_to,
                    prepared_run.topic_id,
                )
            raise

        # Interaction applied 后 Chat 的业务终态已经锁定。后续工作各自结算，
        # 不得再把 Chat 改写为 failed。
        materialization, _ = await asyncio.gather(
            self._dispatch_materialization(
                prepared_run,
                list(payload.materialize_tasks),
            ),
            self._record_retrieval_hits(prepared_run),
        )
        return materialization

    async def _admit_active_interaction(
        self,
        prepared_run: PreparedAgentRun,
        payload: InteractionPayload,
        from_actor: ActorIdentity,
    ) -> InteractionSubmissionReceipt:
        topic_id = prepared_run.topic_id
        correlation = {
            "topic_id": topic_id,
            "agent_id": from_actor.agent_id,
        }

        try:
            receipt = await self._interaction_queue.submit(
                InteractionSubmission(
                    belong_to=prepared_run.belong_to,
                    from_actor=from_actor,
                    interaction_id=prepared_run.interaction_id,
                    payload=payload,
                    requested_topic_id=topic_id,
                    ordering_key=f"topic:{topic_id}",
                    origin="active_chat",
                    correlation=correlation,
                )
            )
        except WorkQueueCapacityError as error:
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_admission",
                reason="capacity_rejected",
            ) from error
        except WorkQueueStoppedError as error:
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_admission",
                reason="queue_stopped",
            ) from error
        except Exception as error:
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_admission",
                reason=type(error).__name__,
            ) from error

        return receipt

    async def _wait_active_interaction(
        self,
        prepared_run: PreparedAgentRun,
        receipt: InteractionSubmissionReceipt,
    ) -> None:
        topic_id = prepared_run.topic_id

        outcome = await self._interaction_queue.wait(receipt)
        if outcome is None:
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_apply",
                reason="outcome_missing",
            )
        if outcome.state != WorkState.SUCCEEDED:
            reason = (
                "queue_stopped"
                if self._interaction_queue.stopped
                and outcome.state in {WorkState.QUEUED, WorkState.RUNNING, WorkState.RETRY_WAIT}
                else f"work_{outcome.state.value}"
            )
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_apply",
                reason=reason,
                work_state=outcome.state,
                error_class=outcome.error_class,
            )
        if outcome.topic_id != topic_id:
            raise ActiveInteractionFinalizationError(
                interaction_id=prepared_run.interaction_id,
                stage="interaction_apply",
                reason="topic_mismatch",
                work_state=outcome.state,
            )

    async def _dispatch_materialization(
        self,
        prepared_run: PreparedAgentRun,
        tasks: list[PendingAtomMaterializeTask],
    ) -> list[MemoryGenerationTask]:
        if not tasks:
            return []

        try:
            return await self._local_bus.request(
                PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
                tasks,
                topic_id=prepared_run.topic_id,
                belong_to=prepared_run.belong_to,
            )
        except Exception as error:
            logger.warning(
                "Active materialization dispatch failed after interaction apply: "
                "interaction_id=%s, error=%s",
                prepared_run.interaction_id,
                type(error).__name__,
                exc_info=True,
            )

            # 这里的调用可能在下游已接纳任务后才断开响应，结果属于 unknown。
            # 只有下游明确返回 rejected 时，才由 Coordinator 按 intent 单独结算失败。
            return []

    def _active_finalization_done(
        self,
        interaction_id: str,
        task: asyncio.Task[list[MemoryGenerationTask]],
    ) -> None:
        if self._active_finalizations.get(interaction_id) is task:
            self._active_finalizations.pop(interaction_id, None)
        self._detached_finalizations.discard(interaction_id)
        if task.cancelled():
            logger.warning("Active finalization continuation cancelled: %s", interaction_id)
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "Active finalization continuation failed: interaction_id=%s, error=%s",
                interaction_id,
                type(error).__name__,
            )

    async def drain_active_finalizations(self) -> None:
        """关闭前等待已经由 Patchouli 接管的 Active continuation。"""

        while True:
            finalizations = [
                task for task in self._active_finalizations.values() if not task.done()
            ]
            if not finalizations:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in finalizations),
                return_exceptions=True,
            )

    async def record_memory_citation(
        self,
        memory_id: str | UUID,
        *,
        identity_scope: IdentityScope,
        source: str = "mtp",
    ) -> Any:
        """记录一次记忆引用事件。"""

        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            normalized_id,
            belong_to=require_identity_scope(identity_scope).workspace_identity,
            source=source,
        )

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
        *,
        identity_scope: IdentityScope,
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题。

        附件租借由持有它的任务进程随进程关闭统一释放，cleanup 不再负责。
        调用方以 prepare 所绑定的 ``resource.search`` 再次授权。本边界
        校验待清理结果的归属，越域结果不会进入补偿流程。
        """
        scope = require_identity_scope(identity_scope)
        if prepared_run.belong_to != scope.workspace_identity:
            # 与 finalize 的同一条件对应：finalize 拒绝提交，cleanup 不越域补偿，但须留下诊断。
            logger.warning(
                "prepared run 的归属与清理授权的 Workspace 不一致，跳过清理: "
                "interaction_id=%s, prepared_workspace=%s, scope_workspace=%s",
                prepared_run.interaction_id,
                prepared_run.belong_to.workspace_id,
                scope.workspace_identity.workspace_id,
            )
            return False
        if not prepared_run.is_new_topic:
            return False
        continuation = self._active_finalizations.get(prepared_run.interaction_id)
        if continuation is not None and not continuation.done():
            logger.info(
                "active finalization continuation 已接管，跳过 prepared topic 清理: %s",
                prepared_run.interaction_id,
            )
            return False
        if await self._interaction_queue.is_accepted(prepared_run.interaction_id):
            logger.info(
                "interaction 已由 submission queue 接管，跳过 prepared topic 清理: %s",
                prepared_run.interaction_id,
            )
            return False
        return await self._cleanup_empty_topic_if_needed(
            prepared_run.belong_to,
            prepared_run.topic_id,
        )

    async def retrieve_for_decision(
        self,
        decision: GatewayDecision,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        enable_retrieval: bool = True,
    ) -> RetrievalResponse:
        """按 GatewayDecision 派生 Patchouli 检索请求。

        检索路由返回完整原子列表（A2 §2.1）；运行上下文仍消费旧协议
        envelope，由本 adapter 构造并测量调用耗时（A2 §2.4，A6 切换）。
        """

        if (
            not enable_retrieval
            or decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        query = RetrievalQuery(
            semantic_query=decision.rewritten_query,
            keywords=list(decision.search_keywords),
            belong_to=belong_to,
            from_actor=from_actor,
        )
        started_at = time.monotonic()
        memories = await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            query,
            top_k=decision.retrieval_plan.top_k,
        )
        return RetrievalResponse.from_memories(
            memories,
            latency_ms=(time.monotonic() - started_at) * 1000,
        )

    async def _record_retrieval_hits(self, prepared_run: PreparedAgentRun) -> None:
        memories = prepared_run.retrieval_result.memories
        seen: set[str] = set()
        for memory in memories:
            memory_id = getattr(memory, "id", None)
            if memory_id is None:
                continue
            memory_key = str(memory_id)
            if memory_key in seen:
                continue
            seen.add(memory_key)
            try:
                await self._local_bus.request(
                    PatchouliLocalRoutes.MEMORY_RECORD_HIT,
                    memory_id,
                    belong_to=prepared_run.belong_to,
                    source="retrieval.finalize",
                )
            except Exception:
                logger.warning(
                    "记录检索命中失败: memory_id=%s",
                    memory_id,
                    exc_info=True,
                )

    async def _cleanup_empty_topic_if_needed(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
    ) -> bool:
        try:
            cleaned = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
                topic_id=topic_id,
                belong_to=belong_to,
            )
            if cleaned:
                logger.info("已清理预创建的空话题: %s", topic_id)
            return cleaned
        except Exception:
            logger.warning("清理预创建空话题失败", exc_info=True)
        return False


__all__ = ["ActiveInteractionFinalizationError", "PatchouliService"]
