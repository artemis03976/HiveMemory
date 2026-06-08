from __future__ import annotations

import logging
import inspect
from typing import TYPE_CHECKING, Any, List
from uuid import UUID

from hivememory.core.models import ActionReducer, Identity, TraceReducer
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AnalyzeAndRetrieveResult,
    InteractionPayload,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.models import (
    PreparedAgentRun,
    StreamPrelude,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.server.models.memory import MemoryResponse
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.control import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.patchouli.eye import TheEye
    from hivememory.patchouli.runtime import PatchouliRuntime

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载记忆域 prepare/finalize/trigger/analyze API。"""

    def __init__(
        self,
        runtime: PatchouliRuntime,
        eye: TheEye,
        global_bus: GlobalSystemBus | None = None,
        local_bus: PatchouliBus | None = None,
    ) -> None:
        self._runtime = runtime
        self._eye = eye
        self._global_bus = global_bus
        self._local_bus = local_bus or runtime.local_bus

    async def analyze_and_retrieve(
        self,
        query: str,
        identity: Identity,
        topic_snapshots: Any = None,
        enable_retrieval: bool = True,
        mode: str = "active",
    ) -> AnalyzeAndRetrieveResult:
        """执行 Patchouli 的标准分析与预检索入口。"""
        gaze_result = await self._eye.gaze(
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )
        retrieval_result = await self.retrieve_for_gaze(
            gaze_result,
            enable_retrieval=enable_retrieval,
            mode=mode,
        )
        return AnalyzeAndRetrieveResult(
            gaze_result=gaze_result,
            retrieval_result=retrieval_result,
        )

    # ========== Phase D: prepare / finalize 公开能力 ==========

    async def prepare_agent_run(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> PreparedAgentRun:
        """
        准备一次 Agent 运行所需的完整记忆上下文。

        执行步骤:
            1. 构造 Identity
            2. 加载 AgentProfile
            3. 获取活跃话题快照
            4. TheEye.gaze — 意图识别 + 查询重写 + 话题路由
            5. prepare_topic — 预创建/刷新话题
            6. retrieve_for_gaze — 预检索
            7. 返回 AgentRunContext 与流式前置信息

        Returns:
            PreparedAgentRun: 顶层可直接用于调用 Alice 的运行上下文
        """
        real_topic_id: str | None = None
        is_new = False

        try:
            identity = Identity(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            agent_profile = await self._require_local_bus().request(
                PatchouliLocalRoutes.GET_AGENT_PROFILE,
                agent_id,
            )
            topic_snapshots = await self._require_local_bus().request(
                PatchouliLocalRoutes.GET_ACTIVE_TOPICS_SNAPSHOTS,
                identity=identity,
            )

            gaze_result = await self._eye.gaze(
                query=user_message,
                topic_snapshots=topic_snapshots,
                identity=identity,
            )

            is_new = gaze_result.target_topic == "NEW_TOPIC"
            real_topic_id, pool_snapshot, topic_context = await self._require_local_bus().request(
                PatchouliLocalRoutes.PREPARE_TOPIC,
                target_topic_id=gaze_result.target_topic,
                new_topic_title=gaze_result.new_topic_title,
                new_topic_summary=gaze_result.new_topic_summary,
                identity=identity,
            )

            retrieval_result = await self.retrieve_for_gaze(
                gaze_result,
                enable_retrieval=enable_memory_retrieval,
            )

            agent_run_context = AgentRunContext(
                identity=identity,
                topic_id=real_topic_id,
                user_message=user_message,
                topic_context=topic_context,
                retrieval_result=retrieval_result,
                agent_profile=agent_profile,
                storage_available=self._runtime.check_storage_health(),
            )

            stream_prelude = StreamPrelude(
                topic_id=real_topic_id,
                is_new_topic=is_new,
                pool_snapshot=pool_snapshot,
                memory_refs=[
                    MemoryResponse.from_atom(m).model_dump(mode="json")
                    for m in retrieval_result.memories
                ],
            )

            return PreparedAgentRun(
                agent_run_context=agent_run_context,
                gaze_result=gaze_result,
                stream_prelude=stream_prelude,
                generation_options=generation_options,
            )
        except Exception:
            if is_new and real_topic_id:
                await self._cleanup_empty_topic_if_needed(real_topic_id)
            raise

    async def finalize_agent_run(
        self,
        prepared_run: PreparedAgentRun,
        loop_result: AgentRunResult,
    ) -> None:
        agent_context = prepared_run.agent_run_context
        gaze_result = prepared_run.gaze_result
        actions = ActionReducer.reduce(loop_result.turn_events)
        mtp_traces = TraceReducer.reduce(actions)

        payload = InteractionPayload(
            user_message=agent_context.user_message,
            mtp_traces=mtp_traces,
            materialize_tasks=loop_result.materialize_tasks,
            identity=agent_context.identity,
            rewritten_query=gaze_result.rewritten_query,
            worth_saving=gaze_result.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
        )

        # 先推 block 进 buffer（被动流），再直驱主动生成，使得当前轮次对话内容被包含在内
        await self._runtime.librarian_core.submit_interaction(
            payload,
            target_topic_id=agent_context.topic_id,
        )

        if loop_result.materialize_tasks:
            await self._runtime.librarian_core.run_active_generation(
                tasks=loop_result.materialize_tasks,
                topic_id=agent_context.topic_id,
            )

        self._record_retrieval_hits(prepared_run)

    # ========== Phase 2: Memory Task API ==========

    def list_memory_tasks(self) -> List[MemoryGenerationTask]:
        return self._runtime.librarian_core.list_tasks()

    def get_memory_task(self, task_id: str) -> MemoryGenerationTask | None:
        return self._runtime.librarian_core.get_task(task_id)

    def cancel_memory_task(self, task_id: str) -> bool:
        return self._runtime.librarian_core.cancel_task(task_id)

    async def record_memory_citation(
        self,
        memory_id: str | UUID,
        source: str = "mtp",
    ) -> Any:
        """Record a lifecycle citation event for a memory atom."""
        lifecycle = getattr(self._runtime.librarian_core, "lifecycle_engine", None)
        if lifecycle is None:
            raise RuntimeError("lifecycle_engine is not available")

        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        result = lifecycle.record_citation(normalized_id, source=source)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
    ) -> bool:
        """清理已 prepare 但未完成 finalize 的流式运行残留。"""
        if not prepared_run.stream_prelude.is_new_topic:
            return False
        return await self._cleanup_empty_topic_if_needed(prepared_run.topic_id)

    async def manual_archive_topic(
        self,
        topic_id: str | None = None,
    ) -> dict[str, Any]:
        """
        手动归档话题 (Archive + Compact)

        用户主动保存当前对话状态。语义为"立即归档 + 生成摘要并保留内存"。
        话题不会被驱逐，可以继续接收新的交互。
        """
        return await self._runtime.librarian_core.manual_archive_topic(topic_id)

    async def evict_topic(self, topic_id: str) -> dict[str, Any]:
        """从活跃话题池中驱逐话题，不归档、不写长期记忆。"""
        buf = self._runtime.librarian_core.perception_layer.buffer_manager.pop_buffer(
            topic_id
        )
        if buf is None:
            return {"success": False, "message": "话题不存在或已被驱逐"}
        return {"success": True, "message": f"话题 {topic_id} 已删除"}

    async def retrieve_for_gaze(
        self,
        gaze_result,
        enable_retrieval: bool = True,
        mode: str = "active",
    ) -> RetrievalResponse:
        if enable_retrieval and gaze_result.intent == GatewayIntent.RAG:
            retrieval_request = RetrievalRequest(
                semantic_query=gaze_result.rewritten_query,
                keywords=gaze_result.search_keywords,
                identity=gaze_result.identity,
            )
            retrieved_result = await self._require_local_bus().request(
                PatchouliLocalRoutes.MEMORY_RETRIEVE,
                retrieval_request,
                mode,
            )
            return retrieved_result

        return RetrievalResponse()

    def _require_global_bus(self) -> GlobalSystemBus:
        if self._global_bus is None:
            raise RuntimeError("PatchouliService 尚未接入 GlobalSystemBus")
        return self._global_bus

    def _require_local_bus(self) -> PatchouliBus:
        if self._local_bus is None:
            raise RuntimeError("PatchouliService 尚未接入 PatchouliBus")
        return self._local_bus

    def _record_retrieval_hits(self, prepared_run: PreparedAgentRun) -> None:
        lifecycle = getattr(self._runtime.librarian_core, "lifecycle_engine", None)
        if lifecycle is None:
            return

        retrieval_result = getattr(prepared_run.agent_run_context, "retrieval_result", None)
        memories = getattr(retrieval_result, "memories", None) or []
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
                lifecycle.record_hit(memory_id, source="retrieval.finalize")
            except Exception:
                logger.warning(
                    "Failed to record retrieval HIT for memory_id=%s",
                    memory_id,
                    exc_info=True,
                )

    async def _cleanup_empty_topic_if_needed(self, topic_id: str) -> bool:
        try:
            buf = self._runtime.librarian_core.perception_layer.get_buffer(topic_id)
            if buf and not buf.blocks:
                self._runtime.librarian_core.perception_layer.swap_out_topic(topic_id)
                logger.info(f"已清理预创建的空话题: {topic_id}")
                return True
        except Exception:
            logger.warning("清理预创建空话题失败", exc_info=True)
        return False
