"""
帕秋莉体系 (The Patchouli System / The Facility)

系统的外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
这是用户（开发者）唯一需要 import 的东西。

架构 (v3.0):
    PatchouliSystem 是"大图书馆"的完整设施 (The Facility)，包含：
    - TheEye (真理之眼): Ingress Gateway，独立于 Kernel 之外
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务

    数据流: User → PatchouliSystem → TheEye.gaze() → Kernel.handle() → Services

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye (Gateway) ──→ PatchouliKernel   │
    │                        ├── Retrieval    │
    │                        ├── Librarian    │
    │                        └── (Koakuma)    │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 3.0
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import InteractionPayload
from hivememory.patchouli.protocol.models import ChatResult

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel import PatchouliKernel
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.worker_agent import WorkerAgentService

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
        self.bus.subscribe(
            "observer.idle_flushed",
            lambda payload: self.kernel.submit_interaction(payload),
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
        """访问检索使魔（代理到 Kernel）"""
        return self.kernel.retrieval_familiar

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问馆长本体（代理到 Kernel）"""
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

    def ingest(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        context: Optional[List[StreamMessage]] = None,
    ) -> Dict[str, Any]:
        """
        被动消息流处理入口 (Passive Observer Mode)

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
            gaze_result, flushed_payload = self.eye.ingest_user(
                content=content, identity=identity, context=context or [],
            )
            if flushed_payload:
                self.kernel.submit_interaction(
                    flushed_payload,
                    target_topic=gaze_result.target_topic,
                )

            # 2. 被动模式检索 (使用 FullContextRenderer 降级渲染)
            hot_result = self.kernel.handle_hot(
                gaze_result,
                mode="passive",
            )

            return {
                "intent": hot_result.intent,
                "rewritten": hot_result.rewritten,
                "keywords": hot_result.keywords,
                "worth_saving": hot_result.worth_saving,
                "memory": hot_result.memory,
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

    def flush_observer_session(
        self,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
    ) -> bool:
        """
        显式 flush 指定 session 的 Observer Buffer (Explicit EOF 触发器)

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
            self.kernel.submit_interaction(payload)
            return True
        return False

    def start_observer_idle_monitor(
        self,
        timeout_seconds: float = 30.0,
        scan_interval_seconds: float = 10.0,
    ) -> None:
        """启动 Observer Buffer 空闲超时监控 (委托给 TheEye，flush 事件通过 bus 路由)"""
        self.eye.start_observer_idle_monitor(
            timeout_seconds=timeout_seconds,
            scan_interval_seconds=scan_interval_seconds,
        )

    def stop_observer_idle_monitor(self) -> None:
        """停止 Observer Buffer 空闲超时监控 (委托给 TheEye)"""
        self.eye.stop_observer_idle_monitor()

    # ========== Kernel 驱动的对话 API ==========

    def chat(
        self,
        user_message: str,
        messages: List[Dict[str, str]],
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        context: Optional[List[StreamMessage]] = None,
        enable_memory_retrieval: bool = True,
    ) -> ChatResult:
        """
        Kernel 驱动的对话入口

        流程:
        1. [The Eye] 意图识别 + 查询重写 (始终执行)
        2. [Kernel.handle_hot] 异步感知投递 + 可选预检索
        3. [Prompt Augmentation] 注入 MTP prompt + 记忆上下文
        4. [The Loop] 递归生成循环 (Phase A→B→C→D)
        5. [Librarian] 异步记录 assistant 回复到感知层

        Args:
            user_message: 当前用户消息 (用于 Eye 分析)
            messages: OpenAI 格式的完整消息列表 (含 system prompt + history)
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            context: 对话历史上下文 (用于 Eye 指代消解)
            enable_memory_retrieval: 是否启用记忆预检索 (False 时 Eye 和感知层仍正常运行，仅跳过检索)

        Returns:
            ChatResult: 递归生成循环的完整结果
        """
        identity = Identity(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )

        # 1. Eye 分析 (始终执行)
        gaze_result = self.eye.gaze(
            query=user_message, context=context or [], identity=identity
        )

        # 2. Kernel 统一管线: 异步感知投递 + 可选预检索
        hot_result = self.kernel.handle_hot(
            gaze_result,
            enable_retrieval=enable_memory_retrieval,
        )

        # 3. 增强 System Prompt (MTP + 记忆上下文)
        messages = [dict(m) for m in messages]  # 浅拷贝
        if messages and messages[0]["role"] == "system":
            mtp_prompt = self.get_mtp_prompt()
            if mtp_prompt:
                messages[0]["content"] += f"\n\n{mtp_prompt}"
            if hot_result.memory:
                messages[0]["content"] += f"\n\n{hot_result.memory}"

        # 4. 递归生成循环
        loop_result = self._recursive_generation_loop(messages, user_id)

        # 5. 构建 InteractionPayload 并提交 (v3.0 统一摄入管道)
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

        # 统一异步提交: 感知层内部处理 URGENT 信号的即时 flush
        self.kernel.submit_interaction(
            payload, target_topic=gaze_result.target_topic
        )

        return loop_result

    def _recursive_generation_loop(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        max_iterations: Optional[int] = None,
    ) -> ChatResult:
        """
        Kernel 递归生成循环 (Phase A→B→C→D)

        Phase A: 调用 WorkerAgent 生成文本 (stop=["⟫"])
        Phase B: 检测是否 MTP 中断
        Phase C: Koakuma 执行 MTP 指令
        Phase D: 将 XML 结果追加到 history，跳回 Phase A

        Args:
            messages: 当前消息列表 (会被原地修改)
            user_id: 用户 ID (用于 Koakuma 权限)
            max_iterations: 最大迭代次数 (默认从配置读取)

        Returns:
            ChatResult: 循环结果
        """
        max_iter = max_iterations or self.config.koakuma.max_recursion_depth
        text_segments: List[str] = []
        mtp_commands: List[str] = []
        iteration = 0

        self.kernel.koakuma.set_current_user(user_id)
        self.kernel.koakuma.reset_interaction_state()

        while iteration < max_iter:
            iteration += 1

            # Phase A: Generate
            result = self._worker_agent.generate(messages)

            # Phase B: Decision
            if not result.was_mtp_interrupted:
                text_segments.append(result.text)
                break

            # MTP 中断 — 累积前缀文本
            text_segments.append(result.prefix_text)

            # Phase C: Execute
            mtp_result = self.kernel.handle_mtp(result.text)

            if mtp_result is None:
                # 误判: stop sequence 命中但无有效 MTP 指令
                text_segments.append(result.mtp_fragment)
                break

            mtp_commands.append(
                mtp_result.command.verb.value
                if mtp_result.command else "UNKNOWN"
            )

            # Phase D: Resume — 构建 fake assistant history
            fake_assistant = result.text + mtp_result.formatted_response
            messages.append({"role": "assistant", "content": fake_assistant})

        return ChatResult(
            final_text="".join(text_segments),
            mtp_iterations=max(0, iteration - 1),
            total_iterations=iteration,
            mtp_commands_executed=mtp_commands,
        )

    def flush_buffer(
        self,
        identity: Identity,
    ) -> None:
        """手动触发感知层 Flush"""
        self.bus.request("perception.flush_buffer", identity=identity)

    def get_buffer_info(
        self,
        identity: Identity,
    ) -> Dict[str, Any]:
        """获取 Buffer 信息"""
        return self.bus.request("perception.get_buffer_info", identity=identity)

    def add_flush_observer(self, observer) -> None:
        """添加 Flush 事件观察者"""
        self.bus.request("librarian.add_flush_observer", observer)

    def get_mtp_prompt(self) -> str:
        """获取 MTP System Prompt 片段（委托给 Kernel）"""
        return self.kernel.get_mtp_prompt()

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
