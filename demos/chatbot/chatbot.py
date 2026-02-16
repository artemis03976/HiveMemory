"""
ChatBot Worker Agent - 与用户对话并将对话流推送给帕秋莉

职责：
1. 管理对话历史（通过 SessionManager）
2. 构建 LLM 消息列表 (system prompt + history + user message)
3. 委托 PatchouliSystem.chat() 执行 Kernel 驱动的递归生成循环

架构定位：
    ChatBotAgent 是"薄壳"应用层，不直接调用 LLM。
    所有 Eye 分析、记忆检索、MTP 递归循环、感知层投递
    均由 PatchouliSystem.chat() 内部完成 (IoC)。

"""

import logging
from typing import List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli import PatchouliSystem
from hivememory.patchouli.protocol.models import ChatResult
from .session_manager import SessionManager, ChatMessage
from .prompts.chatbot import CHATBOT_SYSTEM_PROMPT
from .config import ChatBotConfig, load_chatbot_config

logger = logging.getLogger(__name__)


class ChatBotAgent:
    """
    ChatBot Worker Agent

    薄壳应用层，委托 PatchouliSystem.chat() 执行 Kernel 驱动的对话。
    """

    def __init__(
        self,
        patchouli_system: PatchouliSystem,
        session_manager: SessionManager,
        user_id: str,
        agent_id: str = "chatbot_worker",
        chatbot_config: Optional[ChatBotConfig] = None,
        system_prompt: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        **kwargs,
    ):
        """
        Args:
            patchouli_system: 帕秋莉系统（统一入口）
            session_manager: 会话管理器
            user_id: 用户 ID
            agent_id: Agent ID
            chatbot_config: ChatBot 专用配置
            system_prompt: 系统提示词（可选）
            enable_memory_retrieval: 是否启用记忆检索，默认 True
            **kwargs: 向后兼容 (忽略 config, llm_config 等旧参数)
        """
        self.patchouli_system = patchouli_system
        self.session_manager = session_manager
        self.user_id = user_id
        self.agent_id = agent_id
        self.chatbot_config = chatbot_config or load_chatbot_config()
        self.system_prompt = system_prompt or CHATBOT_SYSTEM_PROMPT
        self.enable_memory_retrieval = enable_memory_retrieval

        # 上一次 chat() 的 ChatResult（用于调试/显示）
        self._last_chat_result: Optional[ChatResult] = None

        logger.info(
            f"ChatBotAgent initialized for user={user_id}, agent={agent_id}, "
            f"memory_retrieval={enable_memory_retrieval}"
        )

    def _build_messages_for_llm(
        self,
        session_id: str,
        new_user_message: str,
        context_window: int = 20,
    ) -> List[Dict[str, str]]:
        """
        构建 LLM API 消息列表 (纯结构，不注入 MTP/记忆)

        MTP prompt 和记忆上下文由 PatchouliSystem.chat() 内部注入。

        Args:
            session_id: 会话 ID
            new_user_message: 新的用户消息
            context_window: 上下文窗口大小

        Returns:
            [{"role": "system", "content": "..."}, ...]
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        history = self.session_manager.get_history(session_id, limit=context_window)
        for msg in history:
            messages.append(msg.to_llm_format())

        messages.append({"role": "user", "content": new_user_message})
        return messages

    def _build_eye_context(
        self,
        session_id: str,
        limit: int = 3,
    ) -> List[StreamMessage]:
        """构建 Eye 指代消解所需的对话上下文"""
        history = self.session_manager.get_history(session_id, limit=limit)
        return [
            StreamMessage(
                message_type=msg.role,
                content=msg.content,
                identity=Identity(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    session_id=session_id,
                ),
            )
            for msg in history
        ]

    def chat(
        self,
        session_id: str,
        user_message: str,
        use_memory: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        处理用户消息并生成回复

        委托 PatchouliSystem.chat() 执行 Kernel 驱动的递归生成循环。
        ChatBotAgent 只负责 Session 管理和消息列表构建。

        Args:
            session_id: 会话 ID
            user_message: 用户消息
            use_memory: 是否使用记忆检索（可选，覆盖默认设置）
            **kwargs: 向后兼容 (忽略 record_to_patchouli 等旧参数)

        Returns:
            AI 助手的回复
        """
        logger.info(f"Processing message for session={session_id}")

        should_use_memory = (
            use_memory if use_memory is not None else self.enable_memory_retrieval
        )

        # 1. 构建消息列表 (system prompt + history + user message)
        messages = self._build_messages_for_llm(session_id, user_message)

        # 2. 构建 Eye 上下文
        context = self._build_eye_context(session_id)

        # 3. 委托给 PatchouliSystem (Kernel 驱动)
        result = self.patchouli_system.chat(
            user_message=user_message,
            messages=messages,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id,
            context=context,
            enable_memory_retrieval=should_use_memory,
        )

        self._last_chat_result = result

        # 4. 保存到 Session
        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", result.final_text)

        logger.info(
            f"Chat completed for session={session_id} "
            f"(iterations={result.total_iterations}, "
            f"mtp={result.mtp_commands_executed})"
        )
        return result.final_text

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
            交互信息字典，包含 MTP 执行统计
        """
        if not self._last_chat_result:
            return None

        return {
            "final_text_len": len(self._last_chat_result.final_text),
            "mtp_iterations": self._last_chat_result.mtp_iterations,
            "total_iterations": self._last_chat_result.total_iterations,
            "mtp_commands_executed": self._last_chat_result.mtp_commands_executed,
        }
