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

import json
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import InteractionPayload, TraceItem
from hivememory.patchouli.protocol.models import ChatResult
from hivememory.patchouli.mtp.models import MTPVerb
from hivememory.infrastructure.trace_context import (
    generate_trace_id, set_trace_context, reset_trace_context
)

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel import PatchouliKernel
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.kernel.runtime.execution_frame import ExecutionFrame
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

        # 0. 创建 SystemBus（系统总线 — 主板）
        from hivememory.infrastructure.system_bus import SystemBus
        self.bus = SystemBus()

        # 1. 初始化 Kernel（内核管理 Retrieval + Librarian + MTP 微服务，注册总线路由）
        self.kernel = PatchouliKernel(config=self.config, bus=self.bus)

        # 2. 初始化 Gateway
        self._init_gateway()

        # 3. 构建 TheEye (通过 bus 访问感知层，Phase 4.5 Agentic Dispatcher)
        self.eye = TheEye(
            engine=self._gateway_engine,
            bus=self.bus,
        )

        # 4. 初始化 Worker Agent (LLM 文本生成引擎)
        self._worker_agent = WorkerAgentService(config=self.config.llm.worker)

        # 5. System 级 Pub/Sub 订阅
        # 注意: 回调中使用 asyncio.create_task 启动异步任务
        self._shutdown_drain_started = False

        async def _on_observer_idle_flushed(payload):
            import asyncio
            asyncio.create_task(self.kernel.submit_interaction(payload))

        self.bus.subscribe(
            "observer.idle_flushed",
            _on_observer_idle_flushed,
        )

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

    # ========== 被动消息流处理 API (Passive Observer Mode) ==========
    # TODO: 参考chat方法重置话题路由时序流
    async def ingest(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        context: Optional[List[StreamMessage]] = None,
    ) -> Dict[str, Any]:
        """
        被动消息流处理入口 (Passive Observer Mode) - 异步版本

        接收外部系统（Discord Bot、微信机器人、传统 Agent 框架）的离散消息，
        通过 TheEye 的 ObserverSessionBuffer 缓冲配对后，构建完整的 InteractionPayload
        提交给感知层进行记忆沉淀。

        与 chat() 的区别:
            - chat(): Kernel 主动驱动 LLM 递归生成循环 (Active/AIOS Mode)
            - ingest(): 被动接收消息，缓冲配对 + Eye 分析 + 检索降级 (Passive Mode)

        数据流 (参考 Passive.md):
            - User: Eye 分析 + 缓冲(自动 flush 上一轮) → 被动模式检索
            - Assistant: 缓冲配对(等待 flush 触发)
            - 其他: 忽略

        Args:
            role: 消息角色 (user/assistant)
            content: 消息内容
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            context: 对话历史上下文 (仅 User 消息需要，用于指代消解)

        Returns:
            Dict: 处理结果
        """
        identity = Identity(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )

        if role == "user":
            # 1. Eye 分析 + 缓冲 (自动 flush 上一轮)
            gaze_result, flushed_payload = await self.eye.ingest_user_async(
                content=content, identity=identity, context=context or [],
            )
            if flushed_payload:
                await self.kernel.submit_interaction(
                    flushed_payload,
                    target_topic=gaze_result.target_topic,
                )

            # 2. 被动模式检索 (使用 FullContextRenderer 降级渲染)
            hot_result = await self.kernel.handle_hot(
                gaze_result,
                mode="passive",
            )

            return {
                "intent": hot_result.intent,
                "rewritten": hot_result.rewritten,
                "keywords": hot_result.keywords,
                "worth_saving": hot_result.worth_saving,
                "memory": hot_result.rendered_memory_context,
            }

        elif role == "assistant":
            self.eye.ingest_assistant(content=content, identity=identity)

            return {
                "intent": "buffered",
                "rewritten": None,
                "keywords": [],
                "worth_saving": True,
                "memory": None,
            }

        else:
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
        payload = self.eye.flush_session(identity)
        if payload:
            await self.kernel.submit_interaction(payload)
            return True
        return False

    def start_observer_idle_monitor(
        self,
        timeout_seconds: float = 30.0,
        scan_interval_seconds: float = 10.0,
        idle_shutdown_seconds: Optional[float] = None,
        lazy_start: bool = False,
    ) -> None:
        """启动 Observer Buffer 空闲超时监控 (委托给 TheEye，flush 事件通过 bus 路由)"""
        self.eye.start_observer_idle_monitor(
            timeout_seconds=timeout_seconds,
            scan_interval_seconds=scan_interval_seconds,
            idle_shutdown_seconds=idle_shutdown_seconds,
            lazy_start=lazy_start,
        )

    def stop_observer_idle_monitor(self) -> None:
        """停止 Observer Buffer 空闲超时监控 (委托给 TheEye)"""
        self.eye.stop_observer_idle_monitor()

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

        self.stop_observer_idle_monitor()

        observer_payloads = self.eye.flush_all_pending_sessions()
        for payload in observer_payloads:
            await self.kernel.submit_interaction(payload)

        perception_result = await self.kernel.librarian_core.perception_layer.flush_all_for_shutdown()
        result = {
            "success": True,
            "observer_payloads_submitted": len(observer_payloads),
            "perception": perception_result,
            "reentrant": False,
        }
        logger.info(
            f"shutdown drain 完成: observer_payloads={len(observer_payloads)}, "
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
            persona = self.kernel.get_agent_persona(agent_id)

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

            # 7. Assemble messages from perception layer context
            messages = self._assemble_messages_from_context(
                topic_context=topic_context,
                hot_result=hot_result,
                user_message=user_message,
                profile=agent_profile,
                persona=persona,
                current_agent_id=agent_id,
            )

            # 8. 递归生成循环
            # 设置 Koakuma 权限沙箱 (Phase 1 多智能体)
            self.kernel.koakuma.set_active_profile(agent_profile)

            loop_result = await self._recursive_generation_loop(
                messages,
                user_id,
                generation_options=generation_options,
                identity=identity,
                agent_profile=agent_profile,
                topic_id=real_topic_id,
            )

            # 9. 构建 InteractionPayload 并提交 (v3.0 统一摄入管道)
            raw_assistant_text = self._reconstruct_raw_assistant_text(messages, loop_result)

            # Koakuma 离线 fallback: 降级为空 traces / None focus
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
                assistant_message=raw_assistant_text,
                mtp_traces=mtp_traces,
                write_focus=write_focus,
                update_focus=update_focus,
                identity=identity,
                rewritten_query=hot_result.rewritten,
                worth_saving=hot_result.worth_saving,
            )

            # 阻塞等待提交完成，确保 token 溢出压缩等操作完成后再返回
            await self.kernel.submit_interaction(
                payload, target_topic=real_topic_id
            )

            logger.info("Chat completed successfully")
            return loop_result

        finally:
            reset_trace_context(tokens)

    async def _recursive_generation_loop(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        max_iterations: Optional[int] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        *,
        identity: Optional[Identity] = None,
        agent_profile=None,
        topic_id: Optional[str] = None,
    ) -> ChatResult:
        """
        帧栈驱动的递归生成循环 (Phase 2 重构)

        支持:
        - 主 Agent 递归 MTP 执行
        - 子 Agent 调用 (CALL 指令 → 帧挂起/恢复)
        - 自动收割 (WRITE/UPDATE 别名跟踪)
        - 黑盒隔离 (子 Agent 细节不污染主 Agent)

        Phase A→B→C→D 循环:
        A. LLM 生成
        B. MTP 拦截检测
        C. MTP 执行 (可能触发 CALL → 子帧派生)
        D. 回填 & 继续

        Args:
            messages: 初始 messages
            user_id: 用户 ID
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            identity: 完整身份标识 (Phase 2)
            agent_profile: 人偶图纸配置 (Phase 2)
            topic_id: 话题 ID (Phase 2)

        Returns:
            ChatResult: 递归生成循环的完整结果
        """
        max_iter = max_iterations or self.config.koakuma.max_recursion_depth

        # 构建身份 (兼容旧接口)
        _identity = identity or Identity(user_id=user_id)

        # 创建主帧
        main_frame = self.kernel.frame_scheduler.create_main_frame(
            agent_profile=agent_profile or self.kernel.load_agent_profile("omni_doll"),
            messages=messages,
            topic_id=topic_id or "",
            identity=_identity,
        )

        # 执行主帧
        return await self._execute_frame(
            frame=main_frame,
            max_iterations=max_iter,
            generation_options=generation_options,
        )

    async def _execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """
        执行单个帧的递归循环

        这是 Phase 2 的核心方法，同时服务于主 Agent 和子 Agent。
        子 Agent 的 CALL 触发递归调用此方法。

        Phase A→B→C→D:
        A. LLM 生成
        B. 自然停止检测
        C. MTP 执行 (SUSPEND → 子帧派生)
        D. 回填 & 继续

        Args:
            frame: 执行帧
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项

        Returns:
            ChatResult: 执行结果
        """
        text_segments: List[str] = []
        mtp_commands: List[str] = []
        iteration = 0

        # 设置 Koakuma 上下文
        self.kernel.koakuma.set_current_identity(frame.identity)
        self.kernel.koakuma.set_active_profile(frame.agent_profile)
        self.kernel.koakuma.set_current_depth(frame.depth)
        self.kernel.koakuma.reset_interaction_state()

        while iteration < max_iterations:
            iteration += 1

            # Phase A: LLM 生成
            result = await self._worker_agent.generate_async(
                frame.working_history,
                **(generation_options or {}),
            )

            # Phase B: 自然停止
            if not result.was_mtp_interrupted:
                text_segments.append(result.text)
                break

            text_segments.append(result.prefix_text)

            # Phase C: MTP 执行
            mtp_result = await self.kernel.handle_mtp(result.text)

            if mtp_result is None:
                text_segments.append(result.mtp_fragment)
                break

            # Phase C.1: 检测 CALL 指令 (SUSPEND 状态)
            if mtp_result.response_status == "suspend":
                ipc_response = await self._handle_call_suspend(
                    frame=frame,
                    mtp_result=mtp_result,
                    assistant_text=result.text,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                )

                # 回填到帧
                frame.working_history.append(
                    {"role": "assistant", "content": result.text + "⟫"}
                )
                frame.working_history.append({
                    "role": "user",
                    "content": f"[System IPC Return]\n{ipc_response}",
                })
                mtp_commands.append("CALL")
                continue

            # Phase C.2: 常规 MTP 指令
            mtp_commands.append(
                mtp_result.command.verb.value
                if mtp_result.command else "UNKNOWN"
            )

            # Phase D: 回填
            frame.working_history.append(
                {"role": "assistant", "content": result.text + "⟫"}
            )
            frame.working_history.append({
                "role": "user",
                "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
            })

            # 自动收割 (仅子帧)
            if frame.is_sub_frame() and mtp_result.command:
                self._try_harvest_alias(frame, mtp_result)

        return ChatResult(
            final_text="".join(text_segments),
            mtp_iterations=max(0, iteration - 1),
            total_iterations=iteration,
            mtp_commands_executed=mtp_commands,
        )

    async def _handle_call_suspend(
        self,
        frame: ExecutionFrame,
        mtp_result,
        assistant_text: str,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        处理 CALL 指令的 SUSPEND 状态

        流程:
        1. 解析 CALL 参数
        2. 挂起当前帧
        3. 派生子帧
        4. 递归执行子帧
        5. 恢复父帧
        6. 组装 IPC 返回 payload

        Args:
            frame: 当前帧 (将被挂起)
            mtp_result: MTP 执行结果 (含 CALL 参数)
            assistant_text: LLM 生成的文本
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项

        Returns:
            str: 格式化的 IPC 返回 payload (XML 格式)
        """
        call_params = json.loads(mtp_result.response_content)
        target_alias = call_params["target_alias"]
        task = call_params["task"]
        context_refs = call_params.get("context_refs", [])

        logger.info(
            f"CALL suspend: target={target_alias}, task='{task[:80]}...'"
        )

        # 1. 挂起当前帧
        self.kernel.frame_scheduler.suspend_frame(frame)

        try:
            # 2. 派生子帧
            sub_frame = await self.kernel.frame_scheduler.fork_sub_frame(
                parent_frame=frame,
                target_alias=target_alias,
                task=task,
                context_refs=context_refs,
            )

            # 3. 递归执行子帧
            sub_result = await self._execute_frame(
                frame=sub_frame,
                max_iterations=max_iterations,
                generation_options=generation_options,
            )

            # 4. 恢复父帧 (弹出栈)
            self.kernel.frame_scheduler.resume_frame()

            # 5. 恢复父帧的 Koakuma 上下文
            self.kernel.koakuma.set_current_identity(frame.identity)
            self.kernel.koakuma.set_active_profile(frame.agent_profile)
            self.kernel.koakuma.set_current_depth(frame.depth)

            # 6. 记录 CALL 到 TraceItem (轨迹折叠)
            self.kernel.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="success",
            ))

            # 7. 组装 IPC 返回 payload
            return self._assemble_ipc_return(
                sub_result=sub_result,
                harvested_aliases=sub_frame.harvested_aliases,
            )

        except Exception as e:
            logger.error(f"Sub-agent execution failed: {e}", exc_info=True)

            # 恢复父帧
            self.kernel.frame_scheduler.resume_frame()
            self.kernel.koakuma.set_current_identity(frame.identity)
            self.kernel.koakuma.set_active_profile(frame.agent_profile)
            self.kernel.koakuma.set_current_depth(frame.depth)

            # 记录失败轨迹
            self.kernel.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="error",
            ))

            return (
                '<mtp_response status="error" type="ipc_return">\n'
                f'[Sub-Agent Error]: The sub-agent "{target_alias}" encountered '
                f'an error and could not complete the task.\n'
                f'Action: Try a different approach or continue without the sub-agent.\n'
                '</mtp_response>'
            )

    def _assemble_ipc_return(
        self,
        sub_result: ChatResult,
        harvested_aliases: List[str],
    ) -> str:
        """
        组装 IPC 返回 payload (XML 格式)

        将子 Agent 的自然语言回复与自动收割的记忆指针混合打包。

        格式:
        <mtp_response status="success" type="ipc_return">
        [Sub-Agent Reply]:
        登录接口的代码已编写完毕。

        [Artifacts Generated / Updated]:
        - mem_login_api_spec (API 接口逻辑代码)
        </mtp_response>

        Args:
            sub_result: 子 Agent 执行结果
            harvested_aliases: 子 Agent 生成的记忆别名列表

        Returns:
            str: 格式化的 IPC 返回 payload
        """
        lines = ['<mtp_response status="success" type="ipc_return">']
        lines.append("[Sub-Agent Reply]:")
        lines.append(sub_result.final_text)

        if harvested_aliases:
            lines.append("")
            lines.append("[Artifacts Generated / Updated]:")
            for alias in harvested_aliases:
                # 尝试获取记忆摘要
                atom = self.kernel.koakuma.atom_cache.get_atom_by_alias(alias)
                if atom and hasattr(atom, 'index') and atom.index.summary:
                    summary = atom.index.summary[:60]
                    lines.append(f"- {alias} ({summary})")
                else:
                    lines.append(f"- {alias}")

        lines.append("</mtp_response>")
        return "\n".join(lines)

    def _try_harvest_alias(self, frame: ExecutionFrame, mtp_result) -> None:
        """
        尝试从 MTP 执行结果中收割别名 (仅子帧)

        当子 Agent 执行 WRITE/UPDATE 时，提取生成的别名并
        添加到帧的 harvested_aliases 列表中。

        Args:
            frame: 当前子帧
            mtp_result: MTP 执行结果
        """
        if not mtp_result.command:
            return

        verb = mtp_result.command.verb
        if verb not in (MTPVerb.WRITE, MTPVerb.UPDATE):
            return

        # UPDATE 的别名可以直接从 target 获取
        if verb == MTPVerb.UPDATE:
            alias = mtp_result.command.target.single_alias
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested UPDATE alias: {alias}")

        # WRITE 使用延迟捕获，别名需要从 Koakuma 获取
        elif verb == MTPVerb.WRITE:
            alias = self.kernel.koakuma.get_last_generated_alias()
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested WRITE alias: {alias}")

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

        try:
            logger.info("Processing user stream message")

            identity = Identity(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )

            # Load agent profile (Phase 1 多智能体)
            agent_profile = self.kernel.load_agent_profile(agent_id)
            persona = self.kernel.get_agent_persona(agent_id)

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
                persona=persona,
                current_agent_id=agent_id,
            )

            # 6. 流式递归生成循环
            max_iter = self.config.koakuma.max_recursion_depth
            text_segments: List[str] = []
            mtp_commands: List[str] = []
            iteration = 0

            self.kernel.koakuma.set_current_identity(identity)
            self.kernel.koakuma.set_active_profile(agent_profile)
            self.kernel.koakuma.set_current_depth(0)  # Phase 2: 主 Agent depth=0
            self.kernel.koakuma.reset_interaction_state()

            while iteration < max_iter:
                iteration += 1
                gen_result = None

                async for chunk in self._worker_agent.generate_stream(
                    messages,
                    **(generation_options or {}),
                ):
                    if chunk.is_final:
                        gen_result = chunk.result
                        break
                    if not chunk.mtp_detected and chunk.delta:
                        yield {"event": "token", "data": {"content": chunk.delta}}

                if gen_result is None:
                    break

                if not gen_result.was_mtp_interrupted:
                    text_segments.append(gen_result.text)
                    break

                # MTP 中断
                text_segments.append(gen_result.prefix_text)

                verb_hint = "UNKNOWN"
                target_hint = ""
                args_hint = {}
                raw_hint = gen_result.mtp_fragment
                try:
                    from hivememory.patchouli.mtp.parser import MTPParser
                    parsed_hint = MTPParser().complete_and_parse(gen_result.text)
                    verb_hint = parsed_hint.verb.value
                    if parsed_hint.target.is_wildcard:
                        target_hint = "*"
                    elif parsed_hint.target.aliases:
                        target_hint = ",".join(parsed_hint.target.aliases)
                    args_hint = dict(parsed_hint.args)
                    raw_hint = parsed_hint.raw_text or raw_hint
                except Exception:
                    pass
                yield {
                    "event": "mtp_start",
                    "data": {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "iteration": iteration,
                    },
                }

                mtp_result = await self.kernel.handle_mtp(gen_result.text)

                if mtp_result is None:
                    text_segments.append(gen_result.mtp_fragment)
                    yield {
                        "event": "mtp_result",
                        "data": {
                            "verb": verb_hint,
                            "target": target_hint,
                            "args": args_hint,
                            "raw_text": raw_hint,
                            "status": "failed",
                            "iteration": iteration,
                        },
                    }
                    break

                # Phase 2: CALL 指令处理 (SUSPEND 状态)
                if mtp_result.response_status == "suspend":
                    call_params = json.loads(mtp_result.response_content)
                    yield {
                        "event": "mtp_result",
                        "data": {
                            "verb": "CALL",
                            "target": call_params["target_alias"],
                            "args": {"task": call_params["task"][:100]},
                            "raw_text": raw_hint,
                            "status": "suspend",
                            "iteration": iteration,
                        },
                    }

                    # 子 Agent 非流式执行 (黑盒原则)
                    # 构建一个临时主帧用于 suspend/resume
                    temp_frame = ExecutionFrame(
                        process_id=f"pid_stream_main_{iteration}",
                        agent_profile=agent_profile,
                        working_history=messages,
                        depth=0,
                        topic_id=real_topic_id,
                        identity=identity,
                    )
                    ipc_response = await self._handle_call_suspend(
                        frame=temp_frame,
                        mtp_result=mtp_result,
                        assistant_text=gen_result.text,
                        max_iterations=max_iter,
                        generation_options=generation_options,
                    )

                    messages.append(
                        {"role": "assistant", "content": gen_result.text + "⟫"}
                    )
                    messages.append({
                        "role": "user",
                        "content": f"[System IPC Return]\n{ipc_response}",
                    })
                    mtp_commands.append("CALL")

                    yield {
                        "event": "mtp_result",
                        "data": {
                            "verb": "CALL",
                            "target": call_params["target_alias"],
                            "args": {"task": call_params["task"][:100]},
                            "raw_text": raw_hint,
                            "status": "success" if '<mtp_response status="success"' in ipc_response else "error",
                            "iteration": iteration,
                        },
                    }
                    continue

                verb = mtp_result.command.verb.value if mtp_result.command else "UNKNOWN"
                mtp_commands.append(verb)
                if mtp_result.command and mtp_result.command.target:
                    if mtp_result.command.target.is_wildcard:
                        target_hint = "*"
                    elif mtp_result.command.target.aliases:
                        target_hint = ",".join(mtp_result.command.target.aliases)
                    else:
                        target_hint = ""
                    args_hint = dict(mtp_result.command.args or {})
                    raw_hint = mtp_result.command.raw_text or raw_hint

                yield {
                    "event": "mtp_result",
                    "data": {
                        "verb": verb,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": mtp_result.response_status,
                        "iteration": iteration,
                    },
                }

                messages.append({"role": "assistant", "content": gen_result.text + "⟫"})
                messages.append({
                    "role": "user",
                    "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
                })

            # 7. 构建最终结果
            loop_result = ChatResult(
                final_text="".join(text_segments),
                mtp_iterations=max(0, iteration - 1),
                total_iterations=iteration,
                mtp_commands_executed=mtp_commands,
            )

            # 8. 提交 InteractionPayload
            raw_assistant_text = self._reconstruct_raw_assistant_text(messages, loop_result)

            try:
                mtp_traces = self.kernel.koakuma.get_interaction_traces()
                write_focus = self.kernel.koakuma.get_write_focus()
                update_focus = self.kernel.koakuma.get_update_focus()
            except Exception:
                mtp_traces = []
                write_focus = None
                update_focus = None

            interaction_payload = InteractionPayload(
                user_message=user_message,
                assistant_message=raw_assistant_text,
                mtp_traces=mtp_traces,
                write_focus=write_focus,
                update_focus=update_focus,
                identity=identity,
                rewritten_query=hot_result.rewritten,
                worth_saving=hot_result.worth_saving,
            )

            await self.kernel.submit_interaction(
                interaction_payload, target_topic=real_topic_id
            )

            logger.info("Stream completed successfully")
            yield {"event": "done", "data": loop_result.model_dump()}

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
        profile=None,  # AgentProfileConfig (Phase 1)
        persona: str = "",
        current_agent_id: str = "omni_doll",
    ) -> List[Dict[str, str]]:
        """
        从感知层上下文组装 LLM messages

        三明治结构 (Phase 1):
        1. System prompt:
           - Top: MTP 协议教学 + 存储降级通知
           - Middle: 灵魂注入 (persona)
           - Bottom: 预检索记忆 + 话题状态
        2. Topic history (from blocks, 含多角色渲染)
        3. Current user message

        Args:
            topic_context: 话题上下文（来自感知层）
            hot_result: Kernel hot path 结果（包含检索到的记忆）
            user_message: 当前用户消息
            profile: 人偶图纸配置（Phase 1 权限过滤）
            persona: 人偶灵魂文本（Phase 1 灵魂注入）
            current_agent_id: 当前活跃 Agent 别名（Phase 1 多角色渲染）

        Returns:
            List[Dict]: OpenAI 格式的 messages
        """
        from hivememory.engines.perception.context_converter import PerceptionContextConverter
        from hivememory.prompts.system_prompt import SystemPromptBuilder

        messages = []

        # 1. Assemble system prompt via SystemPromptBuilder
        language = self.kernel.config.koakuma.mtp_prompt.language if self.kernel.config.koakuma.mtp_prompt else "zh"
        builder = SystemPromptBuilder(language=language)

        # Top: MTP 协议教学
        mtp_prompt = self.kernel.get_mtp_prompt(profile=profile)
        builder.with_mtp_prompt(mtp_prompt)

        # Top: 存储降级通知
        if mtp_prompt and not self.kernel.check_storage_health():
            builder.with_storage_offline_notice()

        # Middle: 灵魂注入
        if profile and persona:
            builder.with_persona(persona)

        # Bottom: 预检索记忆
        builder.with_memory_context(hot_result.rendered_memory_context)

        # Bottom: 话题状态
        builder.with_topic_state(topic_context.get("state_summary", ""))

        system_prompt = builder.build()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Add topic history from blocks (with multi-agent role rendering)
        history_messages = PerceptionContextConverter.blocks_to_messages(
            blocks=topic_context["blocks"],
            include_state_summary=False,  # Already included in system prompt
            current_agent_id=current_agent_id,
        )
        messages.extend(history_messages)

        # 3. Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    # TODO: 检查此逻辑在MTP指令结果通过 role=user 返回的重构后是否需要调整
    @staticmethod
    def _reconstruct_raw_assistant_text(
        messages: List[Dict[str, str]],
        loop_result: ChatResult,
    ) -> str:
        """
        从 messages 历史中重建完整的 assistant 文本 (含 MTP 噪音)

        递归循环中每次 MTP 中断都会追加一条 fake assistant message 到 messages，
        包含 MTP 指令 + XML 响应。最终的 final_text 是纯净文本。
        此方法将所有 assistant 片段拼接为完整的原始文本。

        Args:
            messages: 递归循环结束后的完整消息列表
            loop_result: 循环结果

        Returns:
            str: 包含 MTP 指令和 XML 响应的完整原始 assistant 文本
        """
        # 收集循环中追加的所有 assistant messages
        assistant_parts = []
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_parts.append(msg["content"])

        if assistant_parts:
            return "\n".join(assistant_parts)

        # Fallback: 如果没有 assistant messages (不应发生)，使用 final_text
        return loop_result.final_text


__all__ = [
    "PatchouliSystem",
]
