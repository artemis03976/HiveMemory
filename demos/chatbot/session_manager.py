"""
会话管理器 - 负责对话历史的持久化和检索

支持：
- Redis 存储会话历史
- Session ID 管理
- 自动过期清理（TTL）
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import redis

logger = logging.getLogger(__name__)


class ChatMessage:
    """单条聊天消息"""

    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        """
        Args:
            role: "user" 或 "assistant"
            content: 消息内容
            timestamp: ISO 格式时间戳（可选）
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ChatMessage":
        """从字典创建"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp")
        )

    def to_llm_format(self) -> Dict[str, str]:
        """转换为 LLM API 格式（去除 timestamp）"""
        return {
            "role": self.role,
            "content": self.content
        }


class SessionManager:
    """
    会话管理器

    使用 Redis 存储和管理对话会话历史
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "hivememory:session",
        ttl_days: int = 7
    ):
        """
        Args:
            redis_client: Redis 客户端实例
            key_prefix: Redis key 前缀
            ttl_days: 会话过期时间（天）
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_days * 24 * 3600
        logger.info(f"SessionManager initialized with prefix={key_prefix}, ttl={ttl_days}d")

    def _get_key(self, session_id: str) -> str:
        """生成 Redis key"""
        return f"{self.key_prefix}:{session_id}:history"

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        获取会话历史

        Args:
            session_id: 会话 ID
            limit: 最多返回多少条（从最新开始，None=全部）

        Returns:
            ChatMessage 列表（按时间正序）
        """
        key = self._get_key(session_id)

        try:
            # 从 Redis 读取
            data = self.redis.get(key)
            if not data:
                logger.debug(f"Session {session_id} not found, returning empty history")
                return []

            # 解析 JSON
            messages_data = json.loads(data)
            messages = [ChatMessage.from_dict(msg) for msg in messages_data]

            # 应用 limit（取最后 N 条）
            if limit and limit > 0:
                messages = messages[-limit:]

            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages

        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {e}")
            return []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """
        添加消息到会话历史

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        key = self._get_key(session_id)

        try:
            # 创建消息对象
            message = ChatMessage(role=role, content=content)

            # 读取现有历史
            existing_messages = self.get_history(session_id)

            # 追加新消息
            existing_messages.append(message)

            # 转换为字典列表
            messages_data = [msg.to_dict() for msg in existing_messages]

            # 写入 Redis
            self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(messages_data, ensure_ascii=False)
            )

            logger.debug(f"Added {role} message to session {session_id}")

        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            raise

    def clear_session(self, session_id: str) -> None:
        """
        清空会话历史

        Args:
            session_id: 会话 ID
        """
        key = self._get_key(session_id)

        try:
            self.redis.delete(key)
            logger.info(f"Cleared session {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            raise

    def get_message_count(self, session_id: str) -> int:
        """
        获取会话消息数量

        Args:
            session_id: 会话 ID

        Returns:
            消息数量
        """
        return len(self.get_history(session_id))

    def session_exists(self, session_id: str) -> bool:
        """
        检查会话是否存在

        Args:
            session_id: 会话 ID

        Returns:
            是否存在
        """
        key = self._get_key(session_id)
        return self.redis.exists(key) > 0

    def extend_ttl(self, session_id: str) -> None:
        """
        延长会话过期时间

        Args:
            session_id: 会话 ID
        """
        key = self._get_key(session_id)

        try:
            self.redis.expire(key, self.ttl_seconds)
            logger.debug(f"Extended TTL for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to extend TTL for session {session_id}: {e}")
