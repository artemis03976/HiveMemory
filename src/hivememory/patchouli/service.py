from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from hivememory.core.models import ActionReducer, Identity, MemoryAtom, TraceReducer
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
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
from hivememory.system.config import MemoryCompilerConfig

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载 Agent prepare/finalize 与交互提交。"""

    def __init__(
        self,
        bus: PatchouliBus,
        *,
        memory_compiler_config: MemoryCompilerConfig | None = None,
    ) -> None:
        self._local_bus = bus
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._compiler = MemoryCompiler()

    async def prepare_agent_run(
        self,
        user_message: str,
        user_id: str,
        *,
        gateway_decision: GatewayDecision,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> PreparedAgentRun:
        """根据 GatewayDecision 准备一次完整的 Agent 运行上下文。"""

        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        real_topic_id: str | None = None
        is_new = gateway_decision.target_topic_id == "NEW_TOPIC"

        try:
            agent_profile = await self._local_bus.request(
                PatchouliLocalRoutes.GET_AGENT_PROFILE,
                agent_id,
            )
            real_topic_id = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_PREPARE,
                target_topic_id=gateway_decision.target_topic_id,
                new_topic_title=gateway_decision.new_topic_title,
                new_topic_summary=gateway_decision.new_topic_summary,
                identity=identity,
            )
            pool_topics = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
                identity=identity,
                include_empty=True,
            )
            topic_context = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_GET,
                real_topic_id,
            )

            retrieval_result = await self.retrieve_for_decision(
                gateway_decision,
                identity=identity,
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
                identity=identity,
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
                memory_refs=[
                    _memory_ref_from_atom(memory)
                    for memory in retrieval_result.memories
                ],
            )

            return PreparedAgentRun(
                agent_run_context=agent_run_context,
                gateway_decision=gateway_decision,
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
    ) -> list[MemoryGenerationTask]:
        """提交交互、触发主动记忆生成并记录检索命中。"""

        agent_context = prepared_run.agent_run_context
        decision = prepared_run.gateway_decision
        actions = ActionReducer.reduce(loop_result.turn_events)
        mtp_traces = TraceReducer.reduce(actions)
        payload = InteractionPayload(
            user_message=agent_context.user_message,
            mtp_traces=mtp_traces,
            materialize_tasks=loop_result.materialize_tasks,
            identity=agent_context.identity,
            rewritten_query=decision.rewritten_query,
            worth_saving=decision.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
            model_used=loop_result.model_used,
        )

        await self._local_bus.request(
            PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION,
            payload,
            target_topic_id=agent_context.topic_id,
        )

        memory_tasks: list[MemoryGenerationTask] = []
        if loop_result.materialize_tasks:
            memory_tasks = await self._local_bus.request(
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
        return await self._local_bus.request(
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
        return await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            normalized_id,
            source=source,
        )

    async def cleanup_prepared_agent_run(
        self,
        prepared_run: PreparedAgentRun,
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题。"""

        if not prepared_run.stream_prelude.is_new_topic:
            return False
        return await self._cleanup_empty_topic_if_needed(prepared_run.topic_id)

    async def retrieve_for_decision(
        self,
        decision: GatewayDecision,
        *,
        identity: Identity,
        enable_retrieval: bool = True,
    ) -> RetrievalResponse:
        """按 GatewayDecision 派生 Patchouli 检索请求。"""

        if (
            not enable_retrieval
            or decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        retrieval_request = RetrievalRequest(
            semantic_query=decision.rewritten_query,
            keywords=list(decision.search_keywords),
            identity=identity,
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
                    source="retrieval.finalize",
                )
            except Exception:
                logger.warning(
                    "记录检索命中失败: memory_id=%s",
                    memory_id,
                    exc_info=True,
                )

    async def _cleanup_empty_topic_if_needed(self, topic_id: str) -> bool:
        try:
            cleaned = await self._local_bus.request(
                PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
                topic_id,
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
        "memory_type": (
            memory_type.value if hasattr(memory_type, "value") else str(memory_type)
        ),
        "tags": list(memory.index.tags),
        "alias": memory.index.alias,
        "content": memory.payload.content,
        "created_at": memory.meta.created_at,
        "updated_at": memory.meta.updated_at,
        "confidence_score": memory.meta.confidence_score,
        "vitality_score": memory.meta.vitality_score,
        "user_id": memory.meta.user_id,
        "access_count": memory.meta.access_count,
    }


__all__ = ["PatchouliService"]
