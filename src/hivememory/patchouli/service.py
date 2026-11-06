from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any
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
    MemoryAtom,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    RetrievalMode,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.patchouli.contracts.errors import ActiveInteractionFinalizationError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    InteractionSubmissionReceipt,
    topic_ordering_key,
)
from hivememory.patchouli.runtime.bus import PatchouliBus

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载 Agent prepare/finalize 与交互接纳编排。

    prepare 只做 Topic 与检索：Profile 解析、附件租借与记忆/附件编译由
    chat 任务进程在 CPU 分配时完成，不属于 Patchouli 公开路由。

    访问边界：本门面是授权点以下的资源 owner，阶段路由只接收授权点组装的
    ``IdentityScope``，不接收工作 actor，也不做 operation 检查；资源归属仍由
    本层与领域实现独立校验。scope 只提供归属：检索的可见性取自单独的检索
    scope，交互的发起者取自显式的参与者。
    """

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        interaction_queue: InteractionSubmissionQueue,
    ) -> None:
        if interaction_queue is None:
            raise TypeError("interaction_queue is required")
        self._local_bus = bus
        self._interaction_queue = interaction_queue
        self._active_finalizations: dict[
            str,
            asyncio.Task[None],
        ] = {}
        self._detached_finalizations: set[str] = set()

    async def prepare_agent_run(
        self,
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        gateway_decision: GatewayDecision,
        retrieval_scope: IdentityScope | None = None,
        enable_memory_retrieval: bool = True,
    ) -> PreparedAgentRun:
        """准备本轮的 Topic 与未编译检索结果（prepare 只做 Topic 与检索）。

        返回的 PreparedAgentRun 携带话题准备结果与检索到的原始记忆原子；
        Profile 解析、附件租借与编译由任务进程在 CPU 分配时完成，用户消息
        与 Gateway 决定也由进程持有。prepare 失败时只清理本轮可能预创建的
        空话题。

        ``identity_scope`` 是系统执行话题准备的 root scope，归属取自它；
        ``retrieval_scope`` 是执行者经 ``memory.search`` 授权得到的 scope，检索
        按它的 actor 的可见性进行，为空时不检索。两份 scope 的 Workspace 不同
        时在任何副作用之前拒绝。
        """
        identity_scope = require_identity_scope(identity_scope)
        belong_to = identity_scope.workspace_identity
        if retrieval_scope is not None and (
            require_identity_scope(retrieval_scope).workspace_identity != belong_to
        ):
            raise WorkspaceMismatchError(details={"interaction_id": interaction_id})
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
            memories: tuple[MemoryAtom, ...] = ()
            if retrieval_scope is not None:
                memories = await self.retrieve_for_decision(
                    gateway_decision,
                    belong_to=belong_to,
                    from_actor=retrieval_scope.actor_identity,
                    enable_retrieval=enable_memory_retrieval,
                )

            return PreparedAgentRun(
                belong_to=belong_to,
                interaction_id=interaction_id,
                topic_id=real_topic_id,
                is_new_topic=is_new,
                memories=memories,
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
        participant: ActorIdentity,
    ) -> None:
        """原样提交封口好的交互记录，并把 post-apply 工作交给 Patchouli 持有。

        ``payload`` 由提交方（任务进程）组装并封口，与 Import Bus 一致；finalize
        原样提交，不改写其内容；附件租借由进程持有并随进程关闭
        释放，finalize 不负责释放。交互提交由系统执行：``identity_scope`` 是
        任务进程调用前经 root 授权得到的 scope，只提供归属；``participant``
        是本轮的执行 Agent，记为交互的发起者。后台 continuation 只携带归属与
        参与者。
        """
        scope = require_identity_scope(identity_scope)
        if prepared_run.belong_to != scope.workspace_identity:
            raise WorkspaceMismatchError(details={"interaction_id": prepared_run.interaction_id})
        continuation = self._active_finalizations.get(prepared_run.interaction_id)
        if continuation is None:
            continuation = asyncio.create_task(
                self._continue_active_finalization(prepared_run, payload, participant),
                name=f"active_finalize_{prepared_run.interaction_id[:8]}",
            )
            self._active_finalizations[prepared_run.interaction_id] = continuation
            continuation.add_done_callback(
                partial(self._active_finalization_done, prepared_run.interaction_id)
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
    ) -> None:
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
        await self._record_retrieval_hits(prepared_run)

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
                    ordering_key=topic_ordering_key(topic_id),
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
                reason="admission_unknown",
                error_class=type(error).__name__,
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

    def _active_finalization_done(
        self,
        interaction_id: str,
        task: asyncio.Task[None],
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

        附件租借由持有它的任务进程随进程关闭统一释放，cleanup 不涉及租借。
        调用方以 root 授权得到的 scope 调用。本边界校验待清理结果的归属，
        越域结果不会进入补偿流程。
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
    ) -> tuple[MemoryAtom, ...]:
        """按 GatewayDecision 派生 Patchouli 检索请求，返回按领域排序的原始记忆。

        检索模式为跳过、要求零条或调用方关闭检索时不发起请求，返回空元组。
        """

        if (
            not enable_retrieval
            or decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return ()

        query = RetrievalQuery(
            semantic_query=decision.rewritten_query,
            keywords=list(decision.search_keywords),
            belong_to=belong_to,
            from_actor=from_actor,
        )
        memories = await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            query,
            top_k=decision.retrieval_plan.top_k,
        )
        return tuple(memories)

    async def _record_retrieval_hits(self, prepared_run: PreparedAgentRun) -> None:
        seen: set[str] = set()
        for memory in prepared_run.memories:
            memory_id = memory.id
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
