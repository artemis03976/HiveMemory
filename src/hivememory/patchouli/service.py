from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from hivememory.core.models import (
    ActionReducer,
    MemoryAtom,
    TraceReducer,
    WorkspaceAccessContext,
    require_workspace_access_context,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    RetrievalMode,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    InteractionPayload,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    InteractionSubmissionReceipt,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.control.pending_atom_settler import PendingAtomSettler
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.config import MemoryCompilerConfig
from hivememory.system.runtime.work_queue import (
    WorkQueueCapacityError,
    WorkQueueStoppedError,
    WorkState,
)

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
    """Patchouli 对外能力门面，承载 Agent prepare/finalize 与交互提交。"""

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        interaction_queue: InteractionSubmissionQueue,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        pending_atom_settler: PendingAtomSettler | None = None,
    ) -> None:
        if interaction_queue is None:
            raise TypeError("interaction_queue is required")
        self._local_bus = bus
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        self._interaction_queue = interaction_queue
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._compiler = MemoryCompiler()
        self._active_finalizations: dict[
            str,
            asyncio.Task[list[MemoryGenerationTask]],
        ] = {}
        self._detached_finalizations: set[str] = set()

    async def prepare_agent_run(
        self,
        user_message: str,
        *,
        access_context: WorkspaceAccessContext,
        gateway_decision: GatewayDecision,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> PreparedAgentRun:
        """根据 GatewayDecision 准备一次完整的 Agent 运行上下文。"""
        access_context = require_workspace_access_context(access_context)
        identity = access_context.actor_identity
        real_topic_id: str | None = None
        is_new = gateway_decision.target_topic_id == "NEW_TOPIC"

        try:
            agent_profile = await self._local_bus.request(
                PatchouliLocalRoutes.GET_AGENT_PROFILE,
                identity.agent_id,
                access_context=access_context,
            )
            real_topic_id = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_PREPARE,
                target_topic_id=gateway_decision.target_topic_id,
                new_topic_title=gateway_decision.new_topic_title,
                new_topic_summary=gateway_decision.new_topic_summary,
                access_context=access_context,
            )
            pool_topics = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                access_context=access_context,
                include_empty=True,
            )
            topic_context = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_GET,
                real_topic_id,
                access_context=access_context,
            )

            retrieval_result = await self.retrieve_for_decision(
                gateway_decision,
                access_context=access_context,
                enable_retrieval=enable_memory_retrieval,
            )
            memory_context = (
                self._compiler.compile(
                    retrieval_result.memories,
                    MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
                    MemoryCompileOptions(
                        retrieval_strategy_config=(
                            self._memory_compiler_config.retrieval_context.strategy
                        ),
                    ),
                ).text
                if retrieval_result.memories
                else ""
            )

            agent_run_context = AgentRunContext(
                access_context=access_context,
                topic_id=real_topic_id,
                user_message=user_message,
                topic_context=topic_context,
                retrieval_result=retrieval_result,
                memory_context=memory_context,
                agent_profile=agent_profile,
                storage_available=await self._local_bus.request(
                    PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
                ),
            )
            stream_prelude = StreamPrelude(
                topic_id=real_topic_id,
                is_new_topic=is_new,
                pool_topics=pool_topics,
                memory_refs=[_memory_ref_from_atom(memory) for memory in retrieval_result.memories],
            )

            return PreparedAgentRun(
                agent_run_context=agent_run_context,
                gateway_decision=gateway_decision,
                stream_prelude=stream_prelude,
                generation_options=generation_options,
            )
        except Exception:
            if is_new and real_topic_id:
                await self._cleanup_empty_topic_if_needed(access_context, real_topic_id)
            raise

    async def finalize_agent_run(
        self,
        prepared_run: PreparedAgentRun,
        loop_result: AgentRunResult,
    ) -> list[MemoryGenerationTask]:
        """提交 interaction，并把 post-apply 工作交给 Patchouli 持有。"""

        agent_context = prepared_run.agent_run_context
        decision = prepared_run.gateway_decision
        actions = ActionReducer.reduce(loop_result.turn_events)
        mtp_traces = TraceReducer.reduce(actions)
        payload = InteractionPayload(
            access_context=prepared_run.access_context,
            user_message=agent_context.user_message,
            mtp_traces=mtp_traces,
            materialize_tasks=loop_result.materialize_tasks,
            rewritten_query=decision.rewritten_query,
            worth_saving=decision.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
            model_used=loop_result.model_used,
        )

        continuation = self._active_finalizations.get(prepared_run.interaction_id)
        if continuation is None:
            continuation = asyncio.create_task(
                self._continue_active_finalization(prepared_run, payload),
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
    ) -> list[MemoryGenerationTask]:
        try:
            receipt = await self._admit_active_interaction(prepared_run, payload)
            await self._wait_active_interaction(prepared_run, receipt)
        except ActiveInteractionFinalizationError as error:
            if (
                prepared_run.stream_prelude.is_new_topic
                and (
                    error.stage == "interaction_apply"
                    or prepared_run.interaction_id in self._detached_finalizations
                )
            ):
                await self._cleanup_empty_topic_if_needed(
                    prepared_run.access_context,
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
    ) -> InteractionSubmissionReceipt:
        topic_id = prepared_run.topic_id
        correlation = {
            "topic_id": topic_id,
            "agent_id": prepared_run.agent_id,
        }
        if prepared_run.identity.session_id:
            correlation["session_id"] = prepared_run.identity.session_id

        try:
            receipt = await self._interaction_queue.submit(
                InteractionSubmission(
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
                access_context=prepared_run.access_context,
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

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        *,
        target_topic_id: str | None = None,
        target_topic: str | None = None,
    ) -> Any:
        return await self._local_bus.request(
            PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION,
            payload,
            target_topic_id=target_topic_id or target_topic or "NEW_TOPIC",
        )

    async def record_memory_citation(
        self,
        memory_id: str | UUID,
        *,
        access_context: WorkspaceAccessContext,
        source: str = "mtp",
    ) -> Any:
        """记录一次记忆引用事件。"""

        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            normalized_id,
            access_context=require_workspace_access_context(access_context),
            source=source,
        )

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题。"""

        if not prepared_run.stream_prelude.is_new_topic:
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
            prepared_run.access_context,
            prepared_run.topic_id,
        )

    async def retrieve_for_decision(
        self,
        decision: GatewayDecision,
        *,
        access_context: WorkspaceAccessContext,
        enable_retrieval: bool = True,
    ) -> RetrievalResponse:
        """按 GatewayDecision 派生 Patchouli 检索请求。"""

        access_context = require_workspace_access_context(access_context)

        if (
            not enable_retrieval
            or decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        retrieval_request = RetrievalRequest(
            semantic_query=decision.rewritten_query,
            keywords=list(decision.search_keywords),
            access_context=access_context,
            top_k=decision.retrieval_plan.top_k,
        )
        return await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            retrieval_request,
        )

    async def _record_retrieval_hits(self, prepared_run: PreparedAgentRun) -> None:
        memories = prepared_run.agent_run_context.retrieval_result.memories
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
                    access_context=prepared_run.access_context,
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
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> bool:
        try:
            cleaned = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
                topic_id,
                access_context=access_context,
            )
            if cleaned:
                logger.info("已清理预创建的空话题: %s", topic_id)
            return cleaned
        except Exception:
            logger.warning("清理预创建空话题失败", exc_info=True)
        return False


def _memory_ref_from_atom(memory: MemoryAtom) -> dict[str, Any]:
    """把 MemoryAtom 投影为前端引用列表使用的扁平结构。"""

    memory_type = memory.index.memory_type
    return {
        "id": str(memory.id),
        "title": memory.index.title,
        "summary": memory.index.summary,
        "memory_type": (memory_type.value if hasattr(memory_type, "value") else str(memory_type)),
        "tags": list(memory.index.tags),
        "alias": memory.index.alias,
        "content": memory.payload.content,
        "created_at": memory.meta.created_at,
        "updated_at": memory.meta.updated_at,
        "confidence_score": memory.meta.confidence_score,
        "vitality_score": memory.meta.vitality_score,
        "user_id": memory.workspace_identity.owner_user_id,
        "access_count": memory.meta.access_count,
    }


__all__ = ["ActiveInteractionFinalizationError", "PatchouliService"]
