from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from hivememory.core.errors import (
    AssetOperationConflictError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    ActionReducer,
    AttachmentSelectionRequest,
    IdentityScope,
    MemoryAtom,
    TraceReducer,
    require_identity_scope,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.core.models.workspace_asset import (
    RepresentationLease,
)
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
from hivememory.engines.attachment_compiler import (
    AttachmentCompiler,
    AttachmentCompileResult,
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
from hivememory.system.runtime.workspace.ports import WorkspaceAssetReaderPort

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
    """Patchouli 对外能力门面，承载 Agent prepare/finalize 与交互接纳编排。"""

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        interaction_queue: InteractionSubmissionQueue,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        pending_atom_settler: PendingAtomSettler | None = None,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        attachment_compiler: AttachmentCompiler | None = None,
    ) -> None:
        if interaction_queue is None:
            raise TypeError("interaction_queue is required")
        self._local_bus = bus
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        # 进程级唯一的 WorkspaceAsset reader（由 assembler 注入）：prepare
        # 边界用它完成附件 resolve/acquire，finalize/cleanup 负责 release。
        self._asset_reader = asset_reader
        # W1-E 附件编译组件：与 MemoryCompiler 职责独立（计划 10.1 节）。
        self._attachment_compiler = attachment_compiler or AttachmentCompiler()
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
        identity_scope: IdentityScope,
        interaction_id: str,
        gateway_decision: GatewayDecision,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        selected_attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> PreparedAgentRun:
        """根据 GatewayDecision 准备一次完整的 Agent 运行上下文。

        ``selected_attachments`` 是 Chat 请求冻结的附件选择：prepare 在
        Patchouli 边界按用户顺序逐项 acquire READY representation 并核对
        版本摘要（计划 9.3 节）；任一项失败时释放已取得的 lease 并拒绝
        整个 run。返回的 PreparedAgentRun 携带有序 lease，正文
        正文拼接与 token 预算属于 W1-E 的 AttachmentCompiler；display name 已在资产注册时
        确定，并由 lease 传递给 compiler。
        """
        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        real_topic_id: str | None = None
        is_new = gateway_decision.target_topic_id == "NEW_TOPIC"
        attachment_leases: list[RepresentationLease] = []

        try:
            agent_profile = await self._local_bus.request(
                PatchouliLocalRoutes.GET_AGENT_PROFILE,
                identity.agent_id,
                identity_scope=identity_scope,
            )
            real_topic_id = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_PREPARE,
                target_topic_id=gateway_decision.target_topic_id,
                new_topic_title=gateway_decision.new_topic_title,
                new_topic_summary=gateway_decision.new_topic_summary,
                identity_scope=identity_scope,
            )
            pool_topics = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                identity_scope=identity_scope,
                include_empty=True,
            )
            topic_context = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_GET,
                real_topic_id,
                identity_scope=identity_scope,
            )

            retrieval_result = await self.retrieve_for_decision(
                gateway_decision,
                identity_scope=identity_scope,
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

            # 附件选择：逐项 acquire READY representation 并核对版本摘要。
            # reader 的同一 Store 临界区已完成 Workspace/ref、asset READY 与
            # representation READY 校验并建立 lease，无需先做 resolve_asset。
            for selection in selected_attachments or []:
                lease = await self._acquire_selected_attachment(identity_scope, selection)
                attachment_leases.append(lease)

            # W1-E：附件正文编译发生在 prepare 阶段，lease 与编译同处一条
            # prepared-run 生命周期；此处即可确定 used_attachments。编译失败
            # （如全部附件无法编译）沿既有 except 路径释放 lease 并拒绝 run。
            attachment_compile_result: AttachmentCompileResult = self._attachment_compiler.compile(
                leases=tuple(attachment_leases),
            )

            agent_run_context = AgentRunContext(
                identity_scope=identity_scope,
                interaction_id=interaction_id,
                topic_id=real_topic_id,
                user_message=user_message,
                topic_context=topic_context,
                retrieval_result=retrieval_result,
                memory_context=memory_context,
                agent_profile=agent_profile,
                storage_available=await self._local_bus.request(
                    PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
                ),
                attachment_compile_result=attachment_compile_result,
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
                attachment_leases=tuple(attachment_leases),
            )
        except Exception:
            # prepare 失败：立即释放本轮已取得的 lease（容忍 Store 关闭），
            # 并沿既有路径清理可能预创建的空话题。
            self._release_attachment_leases(attachment_leases)
            if is_new and real_topic_id:
                await self._cleanup_empty_topic_if_needed(identity_scope, real_topic_id)
            raise

    async def _acquire_selected_attachment(
        self,
        identity_scope: IdentityScope,
        selection: AttachmentSelectionRequest,
    ) -> RepresentationLease:
        """acquire 单个选中附件并核对客户端提供的版本摘要。"""
        if self._asset_reader is None:
            raise WorkspaceDomainError(
                "当前系统未装配附件读取能力，不能处理附件选择",
                details={"reason": "asset_reader_unavailable"},
            )
        lease = self._asset_reader.acquire_ready_representation(
            identity_scope,
            selection.asset_ref,
        )
        representation = lease.representation
        mismatch = (
            lease.asset_ref != selection.asset_ref
            or (
                selection.representation_id is not None
                and representation.representation_id != selection.representation_id
            )
            or (selection.revision is not None and representation.revision != selection.revision)
            or (
                selection.content_hash is not None
                and representation.content_hash != selection.content_hash
            )
        )
        if mismatch:
            self._release_lease(lease)
            raise AssetOperationConflictError(
                "所选附件版本与当前可用表示不一致，请重新选择附件",
                details={
                    "reason": "selection_version_mismatch",
                    "asset_id": representation.asset_id,
                },
            )
        return lease

    def _release_attachment_leases(
        self,
        leases: list[RepresentationLease] | tuple[RepresentationLease, ...],
    ) -> None:
        """逐项释放 lease；Store 关闭等清理错误容忍并记录摘要。"""
        for lease in leases:
            self._release_lease(lease)

    def _release_lease(self, lease: RepresentationLease) -> None:
        """释放单个 lease；幂等语义下重复释放返回 False，不视为错误。"""
        if self._asset_reader is None:
            return
        try:
            self._asset_reader.release_representation_lease(lease.lease_id)
        except WorkspaceDomainError as exc:
            # Store 已关闭等清理路径：记录摘要，不改变已经确定的 Chat 终态。
            logger.warning(
                "释放附件 lease 失败: lease_id=%s, code=%s",
                lease.lease_id,
                exc.code,
            )

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
            user_message=agent_context.user_message,
            mtp_traces=mtp_traces,
            materialize_tasks=loop_result.materialize_tasks,
            rewritten_query=decision.rewritten_query,
            worth_saving=decision.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
            model_used=loop_result.model_used,
            # 从 AttachmentCompileResult 生成一份实际使用引用快照
            used_attachments=(
                list(agent_context.attachment_compile_result.used_attachments)
                if agent_context.attachment_compile_result is not None
                else []
            ),
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
            if prepared_run.stream_prelude.is_new_topic and (
                error.stage == "interaction_apply"
                or prepared_run.interaction_id in self._detached_finalizations
            ):
                await self._cleanup_empty_topic_if_needed(
                    prepared_run.identity_scope,
                    prepared_run.topic_id,
                )
            raise
        finally:
            # 进入 finalize 后 lease 由 finalization continuation 持有：
            # Interaction 与后置工作结束（无论成败）即释放（计划 9.3 节）。
            self._release_attachment_leases(prepared_run.attachment_leases)

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
        prepared_actor = prepared_run.identity_scope.actor_identity
        if prepared_actor.session_id:
            correlation["session_id"] = prepared_actor.session_id

        try:
            receipt = await self._interaction_queue.submit(
                InteractionSubmission(
                    identity_scope=prepared_run.identity_scope,
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
                identity_scope=prepared_run.identity_scope,
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
            identity_scope=require_identity_scope(identity_scope),
            source=source,
        )

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题与附件 lease。

        lease 释放无条件执行：finalize 接管路径由 continuation 自行释放，
        此处的重复释放沿 Store 幂等语义返回 False，不产生副作用。
        """
        self._release_attachment_leases(prepared_run.attachment_leases)

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
            prepared_run.identity_scope,
            prepared_run.topic_id,
        )

    async def retrieve_for_decision(
        self,
        decision: GatewayDecision,
        *,
        identity_scope: IdentityScope,
        enable_retrieval: bool = True,
    ) -> RetrievalResponse:
        """按 GatewayDecision 派生 Patchouli 检索请求。"""

        identity_scope = require_identity_scope(identity_scope)

        if (
            not enable_retrieval
            or decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        retrieval_request = RetrievalRequest(
            semantic_query=decision.rewritten_query,
            keywords=list(decision.search_keywords),
            identity_scope=identity_scope,
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
                    identity_scope=prepared_run.identity_scope,
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
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> bool:
        try:
            cleaned = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
                topic_id,
                identity_scope=identity_scope,
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
        "confidence_score": memory.meta.lifecycle.confidence_score,
        "vitality_score": memory.meta.lifecycle.vitality_score,
        "user_id": memory.workspace_identity.owner_user_id,
        "access_count": memory.meta.lifecycle.access_count,
    }


__all__ = ["ActiveInteractionFinalizationError", "PatchouliService"]
