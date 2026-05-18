from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hivememory.core.models import Identity
from hivememory.core.protocol.models import (
    AnalyzeAndRetrieveResult,
    ChatResult,
    InteractionPayload,
)
from hivememory.patchouli.message_assembler import MessageAssembler
from hivememory.patchouli.models import (
    FinalizeContext,
    PreparedAgentRun,
    StreamPrelude,
)
from hivememory.server.models.memory import MemoryResponse
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

if TYPE_CHECKING:
    from hivememory.patchouli.eye import TheEye
    from hivememory.patchouli.kernel import PatchouliKernel

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载记忆域 prepare/finalize/trigger/analyze API。"""

    def __init__(
        self,
        kernel: PatchouliKernel,
        eye: TheEye,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        self._kernel = kernel
        self._eye = eye
        self._global_bus = global_bus
        self._message_assembler = MessageAssembler(kernel)

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
        hot_result = await self._kernel.handle_hot(
            gaze_result,
            enable_retrieval=enable_retrieval,
            mode=mode,
        )
        return AnalyzeAndRetrieveResult(
            gaze_result=gaze_result,
            hot_result=hot_result,
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
            6. handle_hot — 预检索
            7. 注册预检索别名
            8. 组装 messages

        Returns:
            PreparedAgentRun: 顶层可直接用于调用 Alice 的完整上下文
        """
        real_topic_id: str | None = None
        is_new = False

        try:
            identity = Identity(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            agent_profile = self._kernel.load_agent_profile(agent_id)
            topic_snapshots = await self._kernel.get_topic_snapshots(identity)

            gaze_result = await self._eye.gaze(
                query=user_message,
                topic_snapshots=topic_snapshots,
                identity=identity,
            )

            is_new = gaze_result.target_topic == "NEW_TOPIC"
            real_topic_id, pool_snapshot, topic_context = await self._kernel.prepare_topic(
                target_topic_id=gaze_result.target_topic,
                new_topic_title=gaze_result.new_topic_title,
                new_topic_summary=gaze_result.new_topic_summary,
                identity=identity,
            )

            hot_result = await self._kernel.handle_hot(
                gaze_result,
                enable_retrieval=enable_memory_retrieval,
            )

            if hot_result.retrieved_memories:
                await self._register_preretrieval_aliases(hot_result.retrieved_memories)

            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                current_agent_id=agent_id,
            )

            stream_prelude = StreamPrelude(
                topic_id=real_topic_id,
                is_new_topic=is_new,
                pool_snapshot=pool_snapshot,
                memory_refs=[
                    MemoryResponse.from_atom(m).model_dump(mode="json")
                    for m in hot_result.retrieved_memories
                ],
            )

            finalize_context = FinalizeContext(
                hot_result=hot_result,
                identity=identity,
                topic_id=real_topic_id,
                user_message=user_message,
            )

            return PreparedAgentRun(
                identity=identity,
                agent_id=agent_id,
                topic_id=real_topic_id,
                user_message=user_message,
                messages=messages,
                agent_profile=agent_profile,
                stream_prelude=stream_prelude,
                finalize_context=finalize_context,
                generation_options=generation_options,
            )
        except Exception:
            if is_new and real_topic_id:
                await self._cleanup_empty_topic_if_needed(real_topic_id)
            raise

    async def finalize_agent_run(
        self,
        prepared_run: PreparedAgentRun,
        loop_result: ChatResult,
    ) -> None:
        """
        Agent 运行完成后提交 interaction 并执行后处理。

        Args:
            prepared_run: prepare_agent_run 返回的上下文
            loop_result: Alice 运行返回的 ChatResult
        """
        ctx = prepared_run.finalize_context

        try:
            interaction_state = await self._get_interaction_state()
            mtp_traces = interaction_state["mtp_traces"]
            write_focus = interaction_state["write_focus"]
            update_focus = interaction_state["update_focus"]
        except Exception as e:
            logger.warning(f"Koakuma 离线，降级为空 traces: {e}")
            mtp_traces = []
            write_focus = None
            update_focus = None

        payload = InteractionPayload(
            user_message=ctx.user_message,
            mtp_traces=mtp_traces,
            write_focus=write_focus,
            update_focus=update_focus,
            identity=ctx.identity,
            rewritten_query=ctx.hot_result.rewritten,
            worth_saving=ctx.hot_result.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
        )

        await self._kernel.submit_interaction(payload, target_topic=ctx.topic_id)

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

        Args:
            topic_id: 目标话题 ID。如果为 None，使用最后活跃的话题。

        Returns:
            Dict: 包含 success, topic_id, message, blocks_archived 的结果字典

        Examples:
            >>> # 归档最后活跃话题
            >>> result = await system.manual_archive_topic()

            >>> # 归档指定话题
            >>> result = await system.manual_archive_topic(topic_id="topic_123")
        """
        return await self._kernel.manual_trigger(topic_id)

    def _assemble_messages_from_context(
        self,
        topic_context: dict[str, Any],
        hot_result,
        user_message: str,
        profile=None,
        current_agent_id: str = "omni_doll",
    ) -> list[dict[str, str]]:
        """从感知层上下文组装 LLM messages。"""
        assembler = getattr(self, "_message_assembler", None)
        if assembler is None:
            assembler = MessageAssembler(self._kernel)
            self._message_assembler = assembler

        return assembler.assemble(
            topic_context=topic_context,
            hot_result=hot_result,
            user_message=user_message,
            profile=profile,
            current_agent_id=current_agent_id,
        )

    def _require_global_bus(self) -> GlobalSystemBus:
        if self._global_bus is None:
            raise RuntimeError("PatchouliService 尚未接入 GlobalSystemBus")
        return self._global_bus

    async def _register_preretrieval_aliases(self, memories: Any) -> None:
        await self._require_global_bus().request(
            GlobalRoutes.ALICE_REGISTER_PRERETRIEVAL_ALIASES,
            memories,
        )

    async def _get_interaction_state(self) -> dict[str, Any]:
        return await self._require_global_bus().request(
            GlobalRoutes.ALICE_GET_INTERACTION_STATE,
        )

    async def _cleanup_empty_topic_if_needed(self, topic_id: str) -> bool:
        try:
            buf = self._kernel.librarian_core.perception_layer.get_buffer(topic_id)
            if buf and not buf.blocks:
                self._kernel.librarian_core.perception_layer.swap_out_topic(topic_id)
                logger.info(f"已清理预创建的空话题: {topic_id}")
                return True
        except Exception:
            logger.warning("清理预创建空话题失败", exc_info=True)
        return False
