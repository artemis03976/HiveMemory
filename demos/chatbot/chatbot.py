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

from hivememory.patchouli.config import HiveMemoryConfig
from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.config import LLMConfig
from hivememory.infrastructure.llm import LiteLLMService
from hivememory.patchouli import PatchouliSystem
# SessionManager and ChatMessage are now local (moved to demos/chatbot/)
from .session_manager import SessionManager, ChatMessage
from .prompts.chatbot import CHATBOT_SYSTEM_PROMPT
from .config import ChatBotConfig, load_chatbot_config

logger = logging.getLogger(__name__)


class ChatBotAgent:
    """
    ChatBot Worker Agent

    支持记忆检索的对话机器人

    """

    def __init__(
        self,
        patchouli_system: PatchouliSystem,
        session_manager: SessionManager,
        user_id: str,
        agent_id: str = "chatbot_worker",
        config: Optional[HiveMemoryConfig] = None,
        chatbot_config: Optional[ChatBotConfig] = None,
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        enable_memory_retrieval: bool = True,
    ):
        """
        Args:
            patchouli_system: 帕秋莉系统（统一入口）
            session_manager: 会话管理器
            user_id: 用户 ID
            agent_id: Agent ID
            config: 全局配置对象 (Dependency Injection)
            chatbot_config: ChatBot 专用配置
            llm_config: LLM 配置（model, temperature, max_tokens 等）。如果未提供，尝试从 chatbot_config 获取。
            system_prompt: 系统提示词（可选）
            enable_memory_retrieval: 是否启用记忆检索，默认 True
        """
        self.patchouli_system = patchouli_system
        self.session_manager = session_manager
        self.user_id = user_id
        self.agent_id = agent_id
        self.config = config
        self.chatbot_config = chatbot_config or load_chatbot_config()

        # 解析 LLM 配置并初始化服务
        if llm_config:
            # 确保 llm_config 是对象 (如果是 dict 则转换)
            if isinstance(llm_config, dict):
                self.llm_config = LLMConfig(**llm_config)
            else:
                self.llm_config = llm_config
        else:
            # Fallback default: use ChatBotConfig
            self.llm_config = self.chatbot_config.llm

        # 初始化 LLM 服务
        self.llm_service = LiteLLMService(config=self.llm_config)

        # 默认系统提示词
        self.system_prompt = system_prompt or CHATBOT_SYSTEM_PROMPT

        # 记忆检索相关
        self.enable_memory_retrieval = enable_memory_retrieval

        # 上一次交互的结果（用于调试/显示）
        self._last_interaction_result = None

        logger.info(
            f"ChatBotAgent initialized for user={user_id}, agent={agent_id}, "
            f"memory_retrieval={enable_memory_retrieval}"
        )

    def _retrieve_memory_context(
        self,
        user_message: str,
        session_id: str
    ) -> str:
        """
        检索相关记忆并返回渲染后的上下文

        使用 PatchouliSystem.process_interaction() 统一处理：
            1. 自动进行意图分类和查询重写
            2. 只有当 intent == RAG 时才进行检索
            3. 同时将消息记录到感知层

        Args:
            user_message: 用户当前消息
            session_id: 会话 ID

        Returns:
            渲染后的记忆上下文字符串，如果无相关记忆则返回空字符串
        """
        if not self.enable_memory_retrieval:
            return ""

        try:
            # 获取最近的对话历史作为上下文
            history = self.session_manager.get_history(session_id, limit=3)
            context = [
                StreamMessage(
                    message_type=msg.role,
                    content=msg.content,
                    identity=Identity(
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                        session_id=session_id
                    )
                )
                for msg in history
            ]

            # 调用 PatchouliSystem 统一入口
            result = self.patchouli_system.process_interaction(
                role="user",
                content=user_message,
                user_id=self.user_id,
                agent_id=self.agent_id,
                session_id=session_id,
                context=context,
            )

            # 保存结果供后续使用
            self._last_interaction_result = result

            intent = result.get("intent", "")
            if intent != "RAG":
                logger.debug(
                    f"Intent={intent}, skipping memory retrieval"
                )
                return ""

            memory_context = result.get("memory", "") or ""
            if memory_context:
                logger.info(
                    f"Retrieved memory context for query "
                    f"(rewritten='{result.get('rewritten', '')[:30]}...')"
                )

            return memory_context

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            self._last_interaction_result = None
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

        注意：user 消息已在 _retrieve_memory_context 中通过 process_interaction 处理，
        这里只需要记录 assistant 消息。

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        try:
            # user 消息已在 _retrieve_memory_context 中处理
            # 这里只处理 assistant 消息
            if role == "assistant":
                self.patchouli_system.process_interaction(
                    role=role,
                    content=content,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    session_id=session_id,
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
        1. 检索相关历史记忆（同时将 user 消息记录到感知层）
        2. 从 SessionManager 获取历史对话
        3. 构建包含记忆上下文的 Prompt
        4. 调用 LLM 生成回复
        5. 将对话保存到 SessionManager
        6. （可选）将 assistant 回复推送到感知层

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

        # 1. 检索相关记忆（同时将 user 消息记录到感知层）
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

        # 5. 推送 assistant 回复到感知层
        # 注意：user 消息已在 _retrieve_memory_context 中通过 process_interaction 处理
        if record_to_patchouli:
            self._record_to_buffer(session_id, "assistant", assistant_reply)

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
        identity = Identity(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id
        )
        self.patchouli_system.librarian_core.clear_buffer(identity)

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
        获取上一次交互的信息（用于调试）

        Returns:
            交互信息字典，包含意图、重写查询等
        """
        if not self._last_interaction_result:
            return None

        return {
            "intent": self._last_interaction_result.get("intent"),
            "rewritten": self._last_interaction_result.get("rewritten"),
            "keywords": self._last_interaction_result.get("keywords", []),
            "worth_saving": self._last_interaction_result.get("worth_saving"),
            "has_memory": bool(self._last_interaction_result.get("memory")),
        }
