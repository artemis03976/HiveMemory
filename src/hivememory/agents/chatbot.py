"""
ChatBot Worker Agent - 与用户对话并将对话流推送给帕秋莉

职责：
1. 接收用户消息，调用 LLM 生成回复
2. 管理对话历史（通过 SessionManager）
3. 将对话推送到 ConversationBuffer，触发帕秋莉的记忆生成
4. 支持可配置的 LLM 模型切换

"""

import logging
import uuid
from typing import List, Optional, Dict, Any

import litellm

from hivememory.agents.patchouli import PatchouliAgent
from hivememory.agents.session_manager import SessionManager, ChatMessage
from hivememory.agents.prompts.chatbot import CHATBOT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ChatBotAgent:
    """
    ChatBot Worker Agent

    简单的对话机器人，主要用于测试帕秋莉的记忆生成功能
    """

    def __init__(
        self,
        patchouli: PatchouliAgent,
        session_manager: SessionManager,
        user_id: str,
        agent_id: str = "chatbot_worker",
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Args:
            patchouli: 帕秋莉 Agent（图书管理员）
            session_manager: 会话管理器
            user_id: 用户 ID
            agent_id: Agent ID
            llm_config: LLM 配置（model, temperature, max_tokens 等）
            system_prompt: 系统提示词（可选）
        """
        self.patchouli = patchouli
        self.session_manager = session_manager
        self.user_id = user_id
        self.agent_id = agent_id
        self.llm_config = llm_config or {}

        # 默认系统提示词
        self.system_prompt = system_prompt or CHATBOT_SYSTEM_PROMPT

        logger.info(f"ChatBotAgent initialized for user={user_id}, agent={agent_id}")

    def _build_messages_for_llm(
        self,
        session_id: str,
        new_user_message: str,
        context_window: int = 20
    ) -> List[Dict[str, str]]:
        """
        构建 LLM API 消息列表

        Args:
            session_id: 会话 ID
            new_user_message: 新的用户消息
            context_window: 上下文窗口大小（最多保留多少条历史消息）

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        messages = []

        # 1. 添加系统提示词
        messages.append({
            "role": "system",
            "content": self.system_prompt
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
            # 提取配置
            model = self.llm_config.get("model", "gpt-4o")
            api_key = self.llm_config.get("api_key")
            api_base = self.llm_config.get("api_base")
            temperature = self.llm_config.get("temperature", 0.7)
            max_tokens = self.llm_config.get("max_tokens", 2048)

            # 调用 LiteLLM
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 提取内容
            content = response.choices[0].message.content

            logger.info(f"LLM call successful (model={model}, tokens={response.usage.total_tokens})")
            return content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _record_to_buffer(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """
        将消息记录到 ConversationBuffer（触发帕秋莉）

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        try:
            # 通过 PatchouliAgent 获取或创建 Buffer（全局复用）
            buffer = self.patchouli.get_or_create_buffer(
                user_id=self.user_id,
                agent_id=self.agent_id,
                session_id=session_id
            )
            buffer.add_message(role=role, content=content)
            logger.debug(f"Recorded {role} message to buffer (session={session_id})")
        except Exception as e:
            logger.error(f"Failed to record message to buffer: {e}")
            # 不抛出异常，避免影响对话流

    def chat(
        self,
        session_id: str,
        user_message: str,
        record_to_patchouli: bool = True
    ) -> str:
        """
        处理用户消息并生成回复

        工作流程：
        1. 从 SessionManager 获取历史对话
        2. 调用 LLM 生成回复
        3. 将对话保存到 SessionManager
        4. （可选）将对话推送到 ConversationBuffer（触发帕秋莉）

        Args:
            session_id: 会话 ID
            user_message: 用户消息
            record_to_patchouli: 是否记录到 Buffer（触发帕秋莉），默认 True

        Returns:
            AI 助手的回复

        Raises:
            Exception: LLM 调用失败
        """
        logger.info(f"Processing message for session={session_id}")

        # 1. 构建 LLM 消息
        messages = self._build_messages_for_llm(session_id, user_message)

        # 2. 调用 LLM
        assistant_reply = self._call_llm(messages)

        # 3. 保存到会话历史
        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", assistant_reply)

        # 4. 推送到 ConversationBuffer（帕秋莉监听）
        if record_to_patchouli:
            self._record_to_buffer(session_id, "user", user_message)
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
            "agent_id": self.agent_id
        }
