from __future__ import annotations

import logging
from typing import Any, List
from uuid import UUID

from hivememory.engines.memory_compiler import compile_retrieval_context
from hivememory.core.models import ActionReducer, Identity, TraceReducer
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AnalyzeAndRetrieveResult,
    InteractionPayload,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
from hivememory.server.models.memory import MemoryResponse

from hivememory.patchouli.eye import TheEye

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载 prepare/finalize/trigger/analyze API。"""

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        eye: TheEye,
    ) -> None:
        self._eye = eye
        self._local_bus = bus

    async def analyze_and_retrieve(
        self,
        query: str,
        identity: Identity,
        topic_snapshots: Any = None,
        enable_retrieval: bool = True,
    ) -> AnalyzeAndRetrieveResult:
        """执行 Patchouli 的标准分析与预检索入口。"""
        gaze_result = await self._require_local_bus().request(
            PatchouliLocalRoutes.GATEWAY_GAZE,
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )
        retrieval_result = await self.retrieve_for_gaze(
            gaze_result,
            enable_retrieval=enable_retrieval,
        )
        return AnalyzeAndRetrieveResult(
            gaze_result=gaze_result,
            retrieval_result=retrieval_result,
        )

    # ========== prepare / finalize 公开能力 ==========

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
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                identity=identity,
            )

            gaze_result = await self._require_local_bus().request(
                PatchouliLocalRoutes.GATEWAY_GAZE,
                query=user_message,
                topic_snapshots=topic_snapshots,
                identity=identity,
            )

            is_new = gaze_result.target_topic == "NEW_TOPIC"
            real_topic_id = await self._require_local_bus().request(
                PatchouliLocalRoutes.TOPIC_PREPARE,
                target_topic_id=gaze_result.target_topic,
                new_topic_title=gaze_result.new_topic_title,
                new_topic_summary=gaze_result.new_topic_summary,
                identity=identity,
            )
            pool_topics = await self._require_local_bus().request(
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                identity=identity,
                include_empty=True,
            )
            topic_context = await self._require_local_bus().request(
                PatchouliLocalRoutes.TOPIC_GET,
                real_topic_id,
            )

            retrieval_result = await self.retrieve_for_gaze(
                gaze_result,
                enable_retrieval=enable_memory_retrieval,
            )
            memory_context = compile_retrieval_context(retrieval_result.memories)

            agent_run_context = AgentRunContext(
                identity=identity,
                topic_id=real_topic_id,
                user_message=user_message,
                topic_context=topic_context,
                retrieval_result=retrieval_result,
                memory_context=memory_context,
                agent_profile=agent_profile,
                storage_available=await self._require_local_bus().request(
                    PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
                ),
            )

            stream_prelude = StreamPrelude(
                topic_id=real_topic_id,
                is_new_topic=is_new,
                pool_topics=pool_topics,
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
    ) -> List[MemoryGenerationTask]:
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

        # 先推入短期 buffer，再直接驱动主动生成，确保本轮内容进入生成上下文。
        await self._require_local_bus().request(
            PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION,
            payload,
            target_topic_id=agent_context.topic_id,
        )

        memory_tasks: List[MemoryGenerationTask] = []
        if loop_result.materialize_tasks:
            memory_tasks = await self._require_local_bus().request(
                PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
                loop_result.materialize_tasks,
                topic_id=agent_context.topic_id,
            )

        await self._record_retrieval_hits(prepared_run)
        return memory_tasks

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        *,
        target_topic_id: str | None = None,
        target_topic: str | None = None,
    ) -> Any:
        return await self._require_local_bus().request(
            PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION,
            payload,
            target_topic_id=target_topic_id or target_topic or "NEW_TOPIC",
        )

    async def record_memory_citation(
        self,
        memory_id: str | UUID,
        source: str = "mtp",
    ) -> Any:
        """记录一次记忆引用事件。"""
        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._require_local_bus().request(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            normalized_id,
            source=source,
        )

    async def gaze(self, *, query: str, topic_snapshots: Any, identity: Identity):
        if self._eye is None:
            raise RuntimeError("Patchouli gateway is unavailable")
        return await self._eye.gaze(
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题。"""
        if not prepared_run.stream_prelude.is_new_topic:
            return False
        return await self._cleanup_empty_topic_if_needed(prepared_run.topic_id)

    async def retrieve_for_gaze(
        self,
        gaze_result,
        enable_retrieval: bool = True,
    ) -> RetrievalResponse:
        if enable_retrieval and gaze_result.intent == GatewayIntent.RAG:
            retrieval_request = RetrievalRequest(
                semantic_query=gaze_result.rewritten_query,
                keywords=gaze_result.search_keywords,
                identity=gaze_result.identity,
            )
            return await self._require_local_bus().request(
                PatchouliLocalRoutes.MEMORY_RETRIEVE,
                retrieval_request,
            )

        return RetrievalResponse()

    def _require_local_bus(self) -> PatchouliBus:
        if self._local_bus is None:
            raise RuntimeError("PatchouliService 尚未接入 PatchouliBus")
        return self._local_bus

    async def _record_retrieval_hits(self, prepared_run: PreparedAgentRun) -> None:
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
                await self._require_local_bus().request(
                    PatchouliLocalRoutes.MEMORY_RECORD_HIT,
                    memory_id,
                    source="retrieval.finalize",
                )
            except Exception:
                logger.warning(
                    "Failed to record retrieval HIT for memory_id=%s",
                    memory_id,
                    exc_info=True,
                )

    async def _cleanup_empty_topic_if_needed(self, topic_id: str) -> bool:
        try:
            cleaned = await self._require_local_bus().request(
                PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
                topic_id,
            )
            if cleaned:
                logger.info("已清理预创建的空话题: %s", topic_id)
            return cleaned
        except Exception:
            logger.warning("清理预创建空话题失败", exc_info=True)
        return False
