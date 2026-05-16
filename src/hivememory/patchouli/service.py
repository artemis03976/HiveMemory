from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from hivememory.core.models import Identity
from hivememory.infrastructure.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.patchouli.message_assembler import MessageAssembler
from hivememory.patchouli.protocol.models import (
    AnalyzeAndRetrieveResult,
    ChatResult,
    InteractionPayload,
)
from hivememory.server.models.memory import MemoryResponse
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

if TYPE_CHECKING:
    from hivememory.patchouli.eye import TheEye
    from hivememory.patchouli.kernel import PatchouliKernel

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面，承载 chat / stream / trigger / analyze API。"""

    def __init__(
        self,
        kernel: PatchouliKernel,
        eye: TheEye,
        global_bus: Optional[GlobalSystemBus] = None,
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

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """
        Kernel 驱动的对话入口

        流程:
        1. [Perception Layer] 获取活跃话题快照
        2. [The Eye] 意图识别 + 查询重写 + 话题路由
        3. [Perception Layer] 根据路由决策获取完整话题上下文
        4. [Kernel.handle_hot] 预检索
        5. [Prompt Assembly] 从感知层上下文组装 messages
        6. [The Loop] 递归生成循环 (Phase A→B→C→D)
        7. [Librarian] 异步记录 assistant 回复到感知层

        Args:
            user_message: 当前用户消息
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            enable_memory_retrieval: 是否启用记忆预检索

        Returns:
            ChatResult: 递归生成循环的完整结果
        """
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "PatchouliService.Chat", "foreground")

        try:
            logger.info("Processing user chat message")

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

            real_topic_id, _, topic_context = await self._kernel.prepare_topic(
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
                await self._register_preretrieval_aliases(
                    hot_result.retrieved_memories
                )

            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                current_agent_id=agent_id,
            )

            loop_result = await self._run_agent(
                messages=messages,
                identity=identity,
                agent_id=agent_id,
                topic_id=real_topic_id,
                generation_options=generation_options,
                agent_profile=agent_profile,
            )

            await self._chat_post_process(
                messages=messages,
                loop_result=loop_result,
                hot_result=hot_result,
                identity=identity,
                topic_id=real_topic_id,
                user_message=user_message,
            )

            logger.info("Chat completed successfully")
            return loop_result
        finally:
            reset_trace_context(tokens)

    async def chat_stream(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式对话入口 — chat() 的 SSE 流式变体

        逐 token 推送 LLM 生成文本，MTP 执行过程实时推送状态。
        复用 chat() 的所有私有方法，仅将递归循环改为流式 yield。

        SSE 事件类型:
            - topic_info: 话题路由结果
            - token: LLM 生成的文本增量
            - mtp_start: MTP 指令被拦截
            - mtp_result: MTP 执行完成
            - done: 生成完成
            - error: 错误发生

        Yields:
            Dict[str, Any]: {"event": str, "data": dict}
        """
        trace_id = generate_trace_id("stream")
        tokens = set_trace_context(trace_id, "PatchouliService.Stream", "foreground")

        generation_id = str(uuid.uuid4())
        cancel_event = None

        try:
            logger.info("Processing user stream message")

            yield {
                "event": "generation_id",
                "data": {"generation_id": generation_id},
            }

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

            yield {
                "event": "topic_info",
                "data": {
                    "topic_id": real_topic_id,
                    "is_new": is_new,
                    "pool": pool_snapshot,
                },
            }

            hot_result = await self._kernel.handle_hot(
                gaze_result,
                enable_retrieval=enable_memory_retrieval,
            )

            if hot_result.retrieved_memories:
                await self._register_preretrieval_aliases(
                    hot_result.retrieved_memories
                )

            yield {
                "event": "memory_refs",
                "data": {
                    "memories": [
                        MemoryResponse.from_atom(m).model_dump(mode="json")
                        for m in hot_result.retrieved_memories
                    ],
                },
            }

            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                current_agent_id=agent_id,
            )

            loop_result = None

            stream = await self._run_agent_stream(
                messages=messages,
                identity=identity,
                agent_id=agent_id,
                topic_id=real_topic_id,
                generation_options=generation_options,
                agent_profile=agent_profile,
                cancel_event=cancel_event,
            )
            async for event in stream:
                if event["event"] == "done":
                    loop_result = ChatResult(**event["data"])
                else:
                    yield event

            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            await self._chat_post_process(
                messages=messages,
                loop_result=loop_result,
                hot_result=hot_result,
                identity=identity,
                topic_id=real_topic_id,
                user_message=user_message,
            )

            logger.info("Stream completed successfully")
            yield {
                "event": "done",
                "data": {
                    **loop_result.model_dump(),
                    "stopped": False,
                },
            }
        except Exception as e:
            logger.error(f"chat_stream 异常: {e}", exc_info=True)
            if "is_new" in dir() and is_new and "real_topic_id" in dir():
                try:
                    buf = self._kernel.librarian_core.perception_layer.get_buffer(
                        real_topic_id
                    )
                    if buf and not buf.blocks:
                        self._kernel.librarian_core.perception_layer.swap_out_topic(
                            real_topic_id
                        )
                        logger.info(f"已清理预创建的空话题: {real_topic_id}")
                except Exception:
                    pass
            yield {
                "event": "error",
                "data": {"message": "系统错误，请检查后端服务器"},
            }
        finally:
            reset_trace_context(tokens)

    async def manual_trigger(
        self,
        topic_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        手动触发话题结算 (Archive + Compact)

        用户主动保存当前对话状态。语义为"立即归档 + 生成摘要并保留内存"。
        话题不会被驱逐，可以继续接收新的交互。

        Args:
            topic_id: 目标话题 ID。如果为 None，使用最后活跃的话题。

        Returns:
            Dict: 包含 success, topic_id, message, blocks_archived 的结果字典

        Examples:
            >>> # 触发最后活跃话题
            >>> result = await system.manual_trigger()

            >>> # 触发指定话题
            >>> result = await system.manual_trigger(topic_id="topic_123")
        """
        return await self._kernel.manual_trigger(topic_id)

    def _assemble_messages_from_context(
        self,
        topic_context: Dict[str, Any],
        hot_result,
        user_message: str,
        profile=None,
        current_agent_id: str = "omni_doll",
    ) -> List[Dict[str, str]]:
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

    async def _chat_post_process(
        self,
        messages: List[Dict[str, str]],
        loop_result: ChatResult,
        hot_result,
        identity: Identity,
        topic_id: str,
        user_message: str,
    ) -> None:
        """统一处理 chat/chat_stream 结束后的交互提交。"""
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
            user_message=user_message,
            mtp_traces=mtp_traces,
            write_focus=write_focus,
            update_focus=update_focus,
            identity=identity,
            rewritten_query=hot_result.rewritten,
            worth_saving=hot_result.worth_saving,
            assistant_final_text=loop_result.final_text,
            turn_events=loop_result.turn_events,
        )

        await self._kernel.submit_interaction(payload, target_topic=topic_id)

    def cancel_generation(self, generation_id: str) -> bool:
        """生成取消逻辑在规范化过程中暂时断开。"""
        return False

    def _require_global_bus(self) -> GlobalSystemBus:
        if self._global_bus is None:
            raise RuntimeError("PatchouliService 尚未接入 GlobalSystemBus")
        return self._global_bus

    async def _register_preretrieval_aliases(self, memories: Any) -> None:
        await self._require_global_bus().request(
            GlobalRoutes.ALICE_REGISTER_PRERETRIEVAL_ALIASES,
            memories,
        )

    async def _run_agent(self, **kwargs: Any) -> ChatResult:
        return await self._require_global_bus().request(
            GlobalRoutes.ALICE_RUN_AGENT,
            **kwargs,
        )

    async def _run_agent_stream(
        self,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        return await self._require_global_bus().request(
            GlobalRoutes.ALICE_RUN_AGENT_STREAM,
            **kwargs,
        )

    async def _get_interaction_state(self) -> Dict[str, Any]:
        return await self._require_global_bus().request(
            GlobalRoutes.ALICE_GET_INTERACTION_STATE,
        )
