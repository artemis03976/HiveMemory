"""
ChatBot Worker Agent - 与用户对话并将对话流推送给帕秋莉

职责：
1. 接收用户消息，调用 LLM 生成回复
2. 管理对话历史（通过 SessionManager）
3. 将对话推送到感知层（Perception Layer），触发帕秋莉的记忆生成
4. 支持可配置的 LLM 模型切换
5. 检索历史记忆并注入到对话上下文中

"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Union

from hivememory.core.config import HiveMemoryConfig, LLMConfig
from hivememory.core.llm import get_worker_llm_service, BaseLLMService
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.agents.session_manager import SessionManager, ChatMessage
from hivememory.agents.prompts.chatbot import CHATBOT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ChatBotAgent:
    """
    ChatBot Worker Agent

    支持记忆检索的对话机器人
    
    """

    def __init__(
        self,
        patchouli: PatchouliAgent,
        session_manager: SessionManager,
        user_id: str,
        agent_id: str = "chatbot_worker",
        config: Optional[HiveMemoryConfig] = None,
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        llm_service: Optional[BaseLLMService] = None,
        system_prompt: Optional[str] = None,
        retrieval_engine: Optional[Any] = None,  # RetrievalEngine
        enable_memory_retrieval: bool = True,
        lifecycle_manager: Optional[Any] = None,  # LifecycleManager
        enable_lifecycle_management: bool = True,
    ):
        """
        Args:
            patchouli: 帕秋莉 Agent（图书管理员）
            session_manager: 会话管理器
            user_id: 用户 ID
            agent_id: Agent ID
            config: 全局配置对象 (Dependency Injection)
            llm_config: LLM 配置（model, temperature, max_tokens 等）。如果未提供，尝试从 config 获取。
            llm_service: LLM 服务实例（可选，支持依赖注入）
            system_prompt: 系统提示词（可选）
            retrieval_engine: 记忆检索引擎（可选）
            enable_memory_retrieval: 是否启用记忆检索，默认 True
            lifecycle_manager: 生命周期管理器（可选，Stage 3）
            enable_lifecycle_management: 是否启用生命周期管理，默认 True
        """
        self.patchouli = patchouli
        self.session_manager = session_manager
        self.user_id = user_id
        self.agent_id = agent_id
        self.config = config

        # 解析 LLM 配置和服务
        if llm_service is not None:
            # 优先使用注入的服务
            self.llm_service = llm_service
            if isinstance(llm_config, LLMConfig):
                self.llm_config = llm_config
            else:
                self.llm_config = None
        elif config:
            self.llm_config = config.get_worker_llm_config()
            self.llm_service = get_worker_llm_service(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                api_base=self.llm_config.api_base,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
        elif llm_config:
            # 确保 llm_config 是对象 (如果是 dict 则转换)
            if isinstance(llm_config, dict):
                self.llm_config = LLMConfig(**llm_config)
            else:
                self.llm_config = llm_config
            self.llm_service = get_worker_llm_service(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                api_base=self.llm_config.api_base,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
        else:
            # Fallback default
            from hivememory.core.config import get_worker_llm_config
            self.llm_config = get_worker_llm_config()
            self.llm_service = get_worker_llm_service()

        # 默认系统提示词
        self.system_prompt = system_prompt or CHATBOT_SYSTEM_PROMPT

        # 记忆检索相关
        self.retrieval_engine = retrieval_engine
        self.enable_memory_retrieval = enable_memory_retrieval

        # ========== v2.0 新增: Global Gateway ==========
        self.gateway = None
        try:
            from hivememory.gateway import create_default_gateway
            # 如果 config 有 gateway 配置，传递它
            gateway_config = config.gateway if config and hasattr(config, 'gateway') else None
            self.gateway = create_default_gateway(
                llm_service=self.llm_service,
                config=gateway_config
            )
            logger.info(f"GlobalGateway initialized successfully")
        except ImportError as e:
            logger.warning(f"GlobalGateway not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize GlobalGateway: {e}")

        # 生命周期管理器 (Stage 3)
        self.lifecycle_manager = lifecycle_manager
        self.enable_lifecycle_management = enable_lifecycle_management

        # 上一次检索的结果（用于调试/显示）
        self._last_retrieval_result = None
        # 上一次 Gateway 处理结果（用于感知层）
        self._last_gateway_result = None

        logger.info(
            f"ChatBotAgent initialized for user={user_id}, agent={agent_id}, "
            f"memory_retrieval={enable_memory_retrieval}, "
            f"lifecycle={lifecycle_manager is not None} (enabled={enable_lifecycle_management})"
        )

    def _retrieve_memory_context(
        self,
        user_message: str,
        session_id: str
    ) -> str:
        """
        检索相关记忆并返回渲染后的上下文

        v2.0 更新：集成 Global Gateway
            1. 先调用 Gateway 进行意图分类和查询重写
            2. 只有当 intent == RAG 时才进行检索
            3. 使用 rewritten_query 和 keywords 进行检索

        Args:
            user_message: 用户当前消息
            session_id: 会话 ID

        Returns:
            渲染后的记忆上下文字符串，如果无相关记忆则返回空字符串
        """
        if not self.retrieval_engine or not self.enable_memory_retrieval:
            return ""

        try:
            # ========== v2.0: 先调用 Gateway ==========
            from hivememory.generation.models import ConversationMessage
            from hivememory.gateway.models import GatewayIntent

            # 获取最近的对话历史作为上下文
            history = self.session_manager.get_history(session_id, limit=3)
            context = [
                ConversationMessage(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id,
                    user_id=self.user_id
                )
                for msg in history
            ]

            # 调用 Gateway 进行意图分类和查询重写
            gateway_result = None
            if self.gateway:
                gateway_result = self.gateway.process(
                    query=user_message,
                    context=context
                )
                # 保存结果供后续使用（如 _record_to_buffer）
                self._last_gateway_result = gateway_result

                # 只有当 intent == RAG 时才检索
                if gateway_result.intent != GatewayIntent.RAG:
                    logger.debug(
                        f"Gateway intent={gateway_result.intent.value}, "
                        f"skipping memory retrieval"
                    )
                    return ""

                # 使用 Gateway 重写后的查询和关键词
                query_to_use = gateway_result.content_payload.rewritten_query
                keywords = gateway_result.content_payload.search_keywords
                filters = gateway_result.content_payload.target_filters

                logger.debug(
                    f"Gateway rewrote query: '{user_message[:30]}...' -> "
                    f"'{query_to_use[:30]}...', keywords={keywords}"
                )
            else:
                # 回退到原始行为（没有 Gateway 时）
                query_to_use = user_message
                keywords = None
                filters = None

            # 调用检索引擎
            result = self.retrieval_engine.retrieve_context(
                query=query_to_use,
                user_id=self.user_id,
                context=context,
                keywords=keywords,
                filters=filters,
            )

            # 保存结果用于调试
            self._last_retrieval_result = result

            if result.is_empty():
                logger.debug(f"No relevant memories found for query: '{query_to_use[:30]}...'")
                return ""

            logger.info(
                f"Retrieved {result.memories_count} memories for query "
                f"(latency={result.latency_ms:.1f}ms)"
            )

            return result.rendered_context

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            self._last_retrieval_result = None
            return ""

    def _build_messages_for_llm(
        self,
        session_id: str,
        new_user_message: str,
        context_window: int = 20,
        memory_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        构建 LLM API 消息列表

        Args:
            session_id: 会话 ID
            new_user_message: 新的用户消息
            context_window: 上下文窗口大小（最多保留多少条历史消息）
            memory_context: 记忆上下文字符串（可选）

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        messages = []

        # 1. 添加系统提示词（包含记忆上下文）
        system_content = self.system_prompt
        if memory_context:
            # 将记忆上下文添加到系统提示词末尾
            system_content = f"{self.system_prompt}\n\n{memory_context}"
            logger.debug(f"Injected memory context ({len(memory_context)} chars)")
        
        messages.append({
            "role": "system",
            "content": system_content
        })

        # 2. 添加历史对话（限制数量）
        history = self.session_manager.get_history(session_id, limit=context_window)
        for msg in history:
            messages.append(msg.to_llm_format())

        # 3. 添加当前用户消息
        messages.append({
            "role": "user",
            "content": new_user_message
        })

        logger.debug(f"Built {len(messages)} messages for LLM")
        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 LLM 生成回复

        Args:
            messages: LLM 消息列表

        Returns:
            LLM 生成的回复内容

        Raises:
            Exception: LLM 调用失败
        """
        try:
            content = self.llm_service.complete(messages)
            logger.info("LLM call successful")
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2
    ) -> Optional[str]:
        """
        调用 LLM 生成回复（带重试）

        Args:
            messages: LLM 消息列表
            max_retries: 最大重试次数

        Returns:
            LLM 生成的回复内容，失败时返回 None
        """
        return self.llm_service.complete_with_retry(messages, max_retries=max_retries)

    def _record_to_buffer(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """
        将消息记录到感知层（触发帕秋莉）

        v2.0 更新：传递 Gateway 的 rewritten_query 到感知层

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        try:
            # 使用新的感知层 API
            # PatchouliAgent 会自动管理 Buffer，无需手动获取

            # ========== v2.0: 传递 Gateway 输出 ==========
            # 准备额外的参数
            extra_kwargs = {
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "session_id": session_id
            }

            # 如果有 Gateway 结果且是用户消息，传递 rewritten_query
            if self._last_gateway_result and role == "user":
                extra_kwargs["rewritten_query"] = self._last_gateway_result.content_payload.rewritten_query
                extra_kwargs["gateway_intent"] = self._last_gateway_result.intent.value
                extra_kwargs["worth_saving"] = self._last_gateway_result.memory_signal.worth_saving

            self.patchouli.add_message(
                role=role,
                content=content,
                **extra_kwargs
            )
            logger.debug(f"Recorded {role} message to perception layer (session={session_id})")
        except Exception as e:
            logger.error(f"Failed to record message to perception layer: {e}")
            # 不抛出异常，避免影响对话流

    def chat(
        self,
        session_id: str,
        user_message: str,
        record_to_patchouli: bool = True,
        use_memory: Optional[bool] = None
    ) -> str:
        """
        处理用户消息并生成回复

        工作流程：
        1. **[NEW]** 检索相关历史记忆
        2. 从 SessionManager 获取历史对话
        3. 构建包含记忆上下文的 Prompt
        4. 调用 LLM 生成回复
        5. 将对话保存到 SessionManager
        6. （可选）将对话推送到感知层（触发帕秋莉）

        Args:
            session_id: 会话 ID
            user_message: 用户消息
            record_to_patchouli: 是否记录到 Buffer（触发帕秋莉），默认 True
            use_memory: 是否使用记忆检索（可选，覆盖默认设置）

        Returns:
            AI 助手的回复

        Raises:
            Exception: LLM 调用失败
        """
        logger.info(f"Processing message for session={session_id}")
        
        # 1. 检索相关记忆
        should_use_memory = use_memory if use_memory is not None else self.enable_memory_retrieval
        memory_context = ""
        if should_use_memory:
            memory_context = self._retrieve_memory_context(user_message, session_id)

        # 2. 构建 LLM 消息（包含记忆上下文）
        messages = self._build_messages_for_llm(
            session_id,
            user_message,
            memory_context=memory_context
        )

        # 3. 调用 LLM
        assistant_reply = self._call_llm(messages)

        # 4. 保存到会话历史
        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", assistant_reply)

        # 5. 推送到 ConversationBuffer（帕秋莉监听）
        if record_to_patchouli:
            self._record_to_buffer(session_id, "user", user_message)
            self._record_to_buffer(session_id, "assistant", assistant_reply)
        
        # 6. 更新记忆访问统计（如果使用了记忆）
        if self._last_retrieval_result and not self._last_retrieval_result.is_empty():
            try:
                self.retrieval_engine.update_access_stats(
                    self._last_retrieval_result.memories
                )
            except Exception as e:
                logger.debug(f"Failed to update access stats: {e}")

            # 7. 记录生命周期 HIT 事件 (Stage 3)
            if self.lifecycle_manager and self.enable_lifecycle_management:
                try:
                    for memory in self._last_retrieval_result.memories:
                        self.lifecycle_manager.record_hit(memory.id, source="chatbot")
                    logger.debug(f"Recorded {len(self._last_retrieval_result.memories)} HIT events")
                except Exception as e:
                    logger.debug(f"Failed to record HIT events: {e}")

        logger.info(f"Chat completed for session={session_id}")
        return assistant_reply

    def clear_session(self, session_id: str) -> None:
        """
        清空会话历史

        Args:
            session_id: 会话 ID
        """
        # 清空 SessionManager 中的会话历史
        self.session_manager.clear_session(session_id)

        # 清空 Patchouli 的 Buffer
        self.patchouli.clear_buffer(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id
        )

        logger.info(f"Cleared session {session_id} (including buffer)")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话 ID

        Returns:
            会话信息字典
        """
        return {
            "session_id": session_id,
            "message_count": self.session_manager.get_message_count(session_id),
            "exists": self.session_manager.session_exists(session_id),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "memory_retrieval_enabled": self.enable_memory_retrieval
        }
    
    def get_last_retrieval_info(self) -> Optional[Dict[str, Any]]:
        """
        获取上一次记忆检索的信息（用于调试）

        Returns:
            检索信息字典，包含检索到的记忆数量和延迟
        """
        if not self._last_retrieval_result:
            return None

        result = self._last_retrieval_result
        return {
            "should_retrieve": result.should_retrieve,
            "memories_count": result.memories_count,
            "latency_ms": result.latency_ms,
            "query_used": result.query_used,
            "memories": [
                {
                    "title": m.index.title,
                    "type": m.index.memory_type.value,
                    "confidence": m.meta.confidence_score
                }
                for m in result.memories[:5]  # 最多显示 5 条
            ]
        }

    def record_feedback(self, memory_id, positive: bool = True):
        """
        记录用户反馈到生命周期管理器 (Stage 3)

        Args:
            memory_id: 记忆 ID
            positive: 是否正面反馈，默认 True

        Returns:
            ReinforcementResult 如果成功，None 如果未配置 lifecycle_manager 或已禁用
        """
        if self.lifecycle_manager and self.enable_lifecycle_management:
            return self.lifecycle_manager.record_feedback(memory_id, positive)
        return None

    def get_lifecycle_manager(self):
        """
        获取生命周期管理器 (Stage 3)

        Returns:
            LifecycleManager 实例，如果未配置则返回 None
        """
        return self.lifecycle_manager
