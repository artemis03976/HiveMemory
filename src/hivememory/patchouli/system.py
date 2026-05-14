"""
帕秋莉体系 (The Patchouli System / The Facility)

系统的外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
这是用户（开发者）唯一需要 import 的东西。

架构 (v3.0):
    PatchouliSystem 是"大图书馆"的完整设施 (The Facility)，包含：
    - TheEye (真理之眼): Ingress Gateway，独立于 Kernel 之外
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务

    数据流: User → PatchouliSystem → TheEye.gaze() → Kernel.handle_hot() → Services

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye (Gateway) ──→ PatchouliKernel   │
    │                        ├── Retrieval    │
    │                        ├── Librarian    │
    │                        └── Koakuma      │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 3.0
"""

import asyncio
import logging
import uuid
from typing import AsyncGenerator, List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.message_assembler import MessageAssembler
from hivememory.patchouli.protocol.models import (
    AnalyzeAndRetrieveResult,
    ChatResult,
    InteractionPayload,
)
from hivememory.infrastructure.system_bus import SystemBus
from hivememory.infrastructure.trace_context import (
    generate_trace_id, set_trace_context, reset_trace_context
)

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.passive_ingest import PassiveObserverIngressor, PassiveIngressEvent
from hivememory.patchouli.kernel import PatchouliKernel
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.kernel.runtime.loop_executor import KernelLoopExecutor
from hivememory.patchouli.kernel.runtime.maintenance_scheduler import (
    SystemAsyncScheduler,
    MaintenanceTaskSpec,
)
from hivememory.patchouli.worker_agent import WorkerAgentService
from hivememory.server.models.memory import MemoryResponse

logger = logging.getLogger(__name__)


class PatchouliSystem:
    """
    帕秋莉体系 (The Facility) - HiveMemory 的完整封装 v3.0

    外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
    TheEye 独立于 Kernel 之外，做第一道拦截与信息重整，
    处理完后将标准化请求传入 Kernel 进行调度。

    架构:
        - TheEye (真理之眼): Ingress Gateway，流量入口、意图判断、查询重写
        - PatchouliKernel (帕秋莉内核): 中心调度器
            - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
            - LibrarianCore (馆长本体): 后台记忆维护 (Cold)

    使用示例:
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>>
        >>> system = PatchouliSystem()
        >>>
        >>> result = system.process_interaction(
        ...     role="user",
        ...     content="帮我写贪吃蛇游戏",
        ...     context=[],
        ...     user_id="user123"
        ... )
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
        bus: Optional[SystemBus] = None,
    ):
        """
        初始化帕秋莉系统

        Args:
            config: 完整的 HiveMemory 配置（可选）

        Examples:
            >>> system = PatchouliSystem()
            >>>
            >>> from hivememory.patchouli.config import load_app_config
            >>> config = load_app_config("path/to/config.yaml")
            >>> system = PatchouliSystem(config=config)
        """
        self.config = config or load_app_config()

        # 0. 创建/注入 SystemBus（系统总线 — 主板）
        self.bus = bus or SystemBus()

        # 1. 初始化 Kernel（内核管理 Retrieval + Librarian + MTP 微服务，注册总线路由）
        self.kernel = PatchouliKernel(config=self.config, bus=self.bus)

        # 2. 初始化 Gateway
        self._init_gateway()

        # 3. 构建 TheEye (Phase 4.5 Agentic Dispatcher — 仅保留 gaze 职责)
        self.eye = TheEye(
            engine=self._gateway_engine,
            bus=self.bus,
        )

        # 3.5 被动 ingest 编排器 (Phase P1 — 从 TheEye 独立)
        self._passive_ingressor = PassiveObserverIngressor(
            eye=self.eye,
            bus=self.bus,
        )

        # 4. 初始化 Worker Agent (LLM 文本生成引擎)
        self._worker_agent = WorkerAgentService(config=self.config.llm.worker)

        # 4.5 初始化 Loop Executor (帧执行循环管理器)
        self._loop_executor = KernelLoopExecutor(
            kernel=self.kernel,
            worker_agent=self._worker_agent,
        )
        self._message_assembler = MessageAssembler(self.kernel)

        # 5. System 级 Pub/Sub 订阅
        self._shutdown_drain_started = False

        # 6. 取消注册表：generation_id → asyncio.Event
        self._active_generations: Dict[str, asyncio.Event] = {}

        # 7. 统一维护调度器 (Phase S1)
        sched_config = self.config.scheduler
        self._scheduler = SystemAsyncScheduler(
            tick_seconds=sched_config.tick_seconds,
            shutdown_wait_seconds=sched_config.shutdown_wait_seconds,
        )
        self._setup_maintenance_tasks()

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    def _init_gateway(self) -> None:
        """
        初始化 Gateway 相关基础设施

        Gateway LLM 和 Gateway Engine 属于 TheEye 的依赖，
        独立于 Kernel 管理。
        """
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self._gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        from hivememory.engines.gateway import (
            GatewayEngine,
            BaseInterceptor, create_interceptor,
            BaseSemanticAnalyzer, create_semantic_analyzer,
        )

        config = self.config.gateway

        interceptor: BaseInterceptor = create_interceptor(config.interceptor)

        semantic_analyzer: BaseSemanticAnalyzer = create_semantic_analyzer(
            config.analyzer,
            self._gateway_llm_service
        )

        self._gateway_engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    # ========== 向后兼容属性 ==========

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔"""
        return self.kernel.retrieval_familiar

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问帕秋莉本体"""
        return self.kernel.librarian_core

    @property
    def koakuma(self) -> KoakumaRuntime:
        """访问小恶魔 MTP 运行时服务"""
        return self.kernel.koakuma

    @property
    def storage(self):
        """访问存储层（代理到 Kernel）"""
        return self.kernel.storage

    # ========== 生成取消 API ==========

    def register_generation(self, generation_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._active_generations[generation_id] = event
        return event

    def cancel_generation(self, generation_id: str) -> bool:
        event = self._active_generations.get(generation_id)
        if event:
            event.set()
            return True
        return False

    def unregister_generation(self, generation_id: str) -> None:
        self._active_generations.pop(generation_id, None)

    # ========== 统一维护调度器 (Phase S1) ==========

    @property
    def scheduler(self) -> SystemAsyncScheduler:
        """访问统一维护调度器"""
        return self._scheduler

    def _setup_maintenance_tasks(self) -> None:
        """注册所有维护任务到统一调度器"""
        tasks_config = self.config.scheduler.tasks

        # A. Observer idle flush
        self._passive_ingressor.configure_idle_flush(
            timeout_seconds=tasks_config.observer_idle_flush_timeout_seconds,
            on_flush_callback=self._observer_idle_flush_callback,
        )
        self._scheduler.register(
            MaintenanceTaskSpec(
                name="observer_idle_flush",
                interval_seconds=tasks_config.observer_idle_flush_interval_seconds,
                enabled=tasks_config.enable_observer_idle_flush,
            ),
            self._passive_ingressor.scan_idle_sessions_once,
        )

        # B. Perception idle flush
        self._scheduler.register(
            MaintenanceTaskSpec(
                name="perception_idle_flush",
                interval_seconds=tasks_config.perception_idle_flush_interval_seconds,
                enabled=tasks_config.enable_perception_idle_flush,
            ),
            self.kernel.librarian_core.perception_layer.scan_idle_buffers_once,
        )

    async def _observer_idle_flush_callback(
        self, payload, target_topic=None,
    ) -> None:
        """Observer idle flush 结果直接提交到 Kernel（在主 loop 中执行）"""
        await self.kernel.submit_interaction(payload, target_topic=target_topic)

    def start_scheduler(self) -> None:
        """启动统一维护调度器（在主 asyncio loop 中调用）"""
        if not self.config.scheduler.enabled:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "PatchouliSystem.start_scheduler() 必须在运行中的 asyncio 事件循环内调用"
            ) from exc
        self._scheduler.start()

    async def stop_scheduler(self) -> None:
        """停止统一维护调度器"""
        await self._scheduler.stop()

    async def analyze_and_retrieve(
        self,
        query: str,
        identity: Identity,
        topic_snapshots: Any = None,
        enable_retrieval: bool = True,
        mode: str = "active",
    ) -> AnalyzeAndRetrieveResult:
        """
        执行 Patchouli 的标准分析与预检索入口。

        用于将 TheEye 的入口分析与 Kernel 的预检索整合为一个
        可公开暴露的子系统能力，便于通过总线对接顶层应用服务。
        """
        gaze_result = await self.eye.gaze(
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )
        hot_result = await self.kernel.handle_hot(
            gaze_result,
            enable_retrieval=enable_retrieval,
            mode=mode,
        )
        return AnalyzeAndRetrieveResult(
            gaze_result=gaze_result,
            hot_result=hot_result,
        )

    # ========== 被动消息流处理 API (Passive Observer Mode) ==========

    async def ingest_event(
        self,
        event: PassiveIngressEvent,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        被动消息流处理入口 (Passive Observer Mode) — 统一事件版本

        接收外部系统的离散事件（user / assistant / tool_call / tool_result），
        通过 PassiveObserverIngressor 缓冲配对后，构建完整的 InteractionPayload
        提交给感知层进行记忆沉淀。

        Args:
            event: 统一事件输入模型
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Dict: 处理结果
        """
        identity = Identity(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )
        outcome = await self._passive_ingressor.route_event(
            event=event,
            identity=identity,
        )

        if outcome.flushed:
            flushed_payload, flushed_target_topic = outcome.flushed
            await self.kernel.submit_interaction(
                flushed_payload,
                target_topic=flushed_target_topic,
            )

        if outcome.kind == "user":
            hot_result = await self.kernel.handle_hot(
                outcome.gaze_result,
                mode="passive",
            )

            return {
                "intent": hot_result.intent,
                "rewritten": hot_result.rewritten,
                "keywords": hot_result.keywords,
                "worth_saving": hot_result.worth_saving,
                "memory": hot_result.rendered_memory_context,
            }

        if outcome.kind == "buffered":
            return {
                "intent": "buffered",
                "rewritten": None,
                "keywords": [],
                "worth_saving": True,
                "memory": None,
            }

        return {
            "intent": "ignored",
            "rewritten": None,
            "keywords": [],
            "worth_saving": False,
            "memory": None,
        }

    async def flush_observer_session(
        self,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> bool:
        """
        显式 flush 指定 session 的 Observer Buffer (Explicit EOF 触发器) - 异步版本

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            bool: 是否有数据被 flush
        """
        identity = Identity(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )
        flushed = self._passive_ingressor.flush_session(identity)
        if flushed:
            payload, target_topic = flushed
            await self.kernel.submit_interaction(
                payload, target_topic=target_topic,
            )
            return True
        return False

    async def shutdown_drain(self) -> Dict[str, Any]:
        """服务关闭前排空 observer buffer 并强制归档所有活跃话题。"""
        if self._shutdown_drain_started:
            logger.info("shutdown drain 已执行，跳过重复调用")
            return {
                "success": True,
                "observer_payloads_submitted": 0,
                "perception": {
                    "success": True,
                    "trigger_reason": "shutdown",
                    "flushed_topics": [],
                    "skipped_topics": [],
                    "archived_blocks": 0,
                },
                "reentrant": True,
            }

        self._shutdown_drain_started = True
        logger.info("开始执行 shutdown drain")

        await self.stop_scheduler()

        flushed_rounds = self._passive_ingressor.flush_all_pending_sessions()
        for payload, target_topic in flushed_rounds:
            await self.kernel.submit_interaction(
                payload, target_topic=target_topic,
            )

        perception_result = await self.kernel.librarian_core.perception_layer.flush_all_for_shutdown()
        result = {
            "success": True,
            "observer_payloads_submitted": len(flushed_rounds),
            "perception": perception_result,
            "reentrant": False,
        }
        logger.info(
            f"shutdown drain 完成: observer_payloads={len(flushed_rounds)}, "
            f"flushed_topics={len(perception_result['flushed_topics'])}"
        )
        return result

    # ========== Kernel 驱动的对话 API ==========

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
        # Set trace context for observability
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "PatchouliSystem.Chat", "foreground")

        try:
            logger.info(f"Processing user chat message")

            # 1. Create identity
            identity = Identity(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )

            # 1.5 Load agent profile (Phase 1 多智能体)
            agent_profile = self.kernel.load_agent_profile(agent_id)

            # 2. Get topic snapshots from perception layer
            topic_snapshots = await self.kernel.get_topic_snapshots(identity)

            # 3. Eye 分析 (with topic snapshots for routing and coreference)
            gaze_result = await self.eye.gaze(
                query=user_message,
                topic_snapshots=topic_snapshots,
                identity=identity
            )

            # 4. 预创建话题 + LRU 驱逐（提前到生成之前）
            is_new = (gaze_result.target_topic == "NEW_TOPIC")
            real_topic_id, _, topic_context = await self.kernel.prepare_topic(
                target_topic_id=gaze_result.target_topic,
                new_topic_title=gaze_result.new_topic_title,
                new_topic_summary=gaze_result.new_topic_summary,
                identity=identity,
            )

            # 5. Kernel 统一管线: 预检索
            hot_result = await self.kernel.handle_hot(
                gaze_result,
                enable_retrieval=enable_memory_retrieval,
            )

            # 6. Assemble messages from perception layer context
            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                current_agent_id=agent_id,
            )

            # 7. 递归生成循环
            # 设置 Koakuma 权限沙箱 (Phase 1 多智能体)
            self.kernel.koakuma.set_active_profile(agent_profile)

            loop_result = await self._loop_executor.execute_main_frame(
                messages=messages,
                max_iterations=None,
                generation_options=generation_options,
                agent_profile=agent_profile,
                topic_id=real_topic_id,
                identity=identity,
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

    # ========== 流式对话 API (SSE) ==========

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
        # Set trace context for observability
        trace_id = generate_trace_id("stream")
        tokens = set_trace_context(trace_id, "PatchouliSystem.Stream", "foreground")

        generation_id = str(uuid.uuid4())
        cancel_event = self.register_generation(generation_id)

        try:
            logger.info("Processing user stream message")

            yield {
                "event": "generation_id",
                "data": {"generation_id": generation_id},
            }

            identity = Identity(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )

            # Load agent profile (Phase 1 多智能体)
            agent_profile = self.kernel.load_agent_profile(agent_id)

            # 1. 获取话题快照
            topic_snapshots = await self.kernel.get_topic_snapshots(identity)

            # 2. Eye 分析
            gaze_result = await self.eye.gaze(
                query=user_message,
                topic_snapshots=topic_snapshots,
                identity=identity,
            )

            # 3. 预创建话题 + LRU 驱逐（提前到生成之前）
            is_new = (gaze_result.target_topic == "NEW_TOPIC")
            real_topic_id, pool_snapshot, topic_context = await self.kernel.prepare_topic(
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

            # 4. 预检索
            hot_result = await self.kernel.handle_hot(
                gaze_result, enable_retrieval=enable_memory_retrieval,
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

            # 5. 组装 messages
            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                current_agent_id=agent_id,
            )

            # 6. 流式递归生成循环（委托给 KernelLoopExecutor）
            self.kernel.koakuma.set_active_profile(agent_profile)
            loop_result = None

            async for event in self._loop_executor.execute_main_frame_stream(
                messages=messages,
                max_iterations=None,
                generation_options=generation_options,
                agent_profile=agent_profile,
                topic_id=real_topic_id,
                identity=identity,
                cancel_event=cancel_event,
            ):
                if event["event"] == "done":
                    from hivememory.patchouli.protocol.models import ChatResult
                    loop_result = ChatResult(**event["data"])
                else:
                    yield event

            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            if not cancel_event.is_set():
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
                    "stopped": cancel_event.is_set(),
                },
            }

        except Exception as e:
            logger.error(f"chat_stream 异常: {e}", exc_info=True)
            # 错误恢复：如果预创建了空的新话题，清理它
            if 'is_new' in dir() and is_new and 'real_topic_id' in dir():
                try:
                    buf = self.kernel.librarian_core.perception_layer.get_buffer(real_topic_id)
                    if buf and not buf.blocks:
                        self.kernel.librarian_core.perception_layer.swap_out_topic(real_topic_id)
                        logger.info(f"已清理预创建的空话题: {real_topic_id}")
                except Exception:
                    pass
            yield {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}

        finally:
            self.unregister_generation(generation_id)
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
        return await self.kernel.manual_trigger(topic_id)

    def _assemble_messages_from_context(
        self,
        topic_context: Dict[str, Any],
        hot_result,  # KernelHotResult
        user_message: str,
        profile=None,  # AgentProfile (Phase 1)
        current_agent_id: str = "omni_doll",
    ) -> List[Dict[str, str]]:
        """
        从感知层上下文组装 LLM messages

        三明治结构 (Phase 1):
        1. System prompt:
           - Top: MTP 协议教学 + 存储降级通知
           - Middle: 灵魂注入 (persona from profile)
           - Bottom: 预检索记忆 + 话题状态
        2. Topic history (from blocks, 含多角色渲染)
        3. Current user message

        Args:
            topic_context: 话题上下文（来自感知层）
            hot_result: Kernel hot path 结果（包含检索到的记忆）
            user_message: 当前用户消息
            profile: 人偶图纸配置（Phase 1 权限过滤 + 灵魂注入）
            current_agent_id: 当前活跃 Agent 别名（Phase 1 多角色渲染）

        Returns:
            List[Dict]: OpenAI 格式的 messages
        """
        assembler = getattr(self, "_message_assembler", None)
        if assembler is None:
            # 兼容绕过 __init__ 的单测夹具，按需懒加载组装器。
            assembler = MessageAssembler(self.kernel)
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
        """
        Chat 后处理通用函数

        统一处理:
        1. 重建原始 assistant 文本
        2. 获取 MTP traces 和 focus
        3. 构建 InteractionPayload
        4. 提交到感知层

        Args:
            messages: 递归循环结束后的完整消息列表
            loop_result: 循环结果
            hot_result: KernelHotResult (包含 rewritten 和 worth_saving)
            identity: 身份标识
            topic_id: 话题 ID
            user_message: 用户原始消息

        Returns:
            None: 无返回值
        """
        try:
            mtp_traces = self.kernel.koakuma.get_interaction_traces()
            write_focus = self.kernel.koakuma.get_write_focus()
            update_focus = self.kernel.koakuma.get_update_focus()
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

        await self.kernel.submit_interaction(payload, target_topic=topic_id)


__all__ = [
    "PatchouliSystem",
]
