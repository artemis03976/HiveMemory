"""
HiveMemory 核心数据模型 - 交互与流转领域

定义用户身份标识和在系统中流转的消息模型。
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.constants import DEFAULT_USER_ID, DEFAULT_AGENT_ID, DEFAULT_TEAM_ID
from hivememory.utils.token_estimator import estimate_tokens


class Identity(BaseModel):
    """
    身份标识组合 - 统一管理用户、Agent 两个核心ID

    用于替代散落的 user_id, agent_id 参数，
    提供统一的身份标识和便捷的操作方法。

    注意：session_id 已被移除，其功能被 topic_id 替代。
    话题的生命周期由 PerceptionLayer 的 topic_id 管理。

    Attributes:
        user_id: 用户标识符
        agent_id: Agent 标识符
    """
    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    team_id: Optional[str] = Field(default=DEFAULT_TEAM_ID, description="团队 ID（用于 Workspace 作用域过滤）")
    session_id: Optional[str] = Field(default=None, description="会话 ID（兼容字段）")

    @property
    def buffer_key(self) -> str:
        """生成用于缓冲区的唯一键"""
        return f"{self.user_id}:{self.agent_id}"

    @property
    def is_valid(self) -> bool:
        """检查身份标识是否有效"""
        return bool(self.user_id and self.agent_id)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "agent_id": "chatbot",
                "session_id": "sess_456"
            }
        }
    )


class StreamMessageType(str, Enum):
    """流式消息类型枚举"""
    USER = "user"             # 用户查询
    SYSTEM = "system"         # 系统消息
    ASSISTANT = "assistant"   # 助手消息
    TOOL = "tool"             # 工具输出
    THOUGHT = "thought"       # 思考过程 (Internal)
    TOOL_CALL = "tool_call"   # 工具调用 (Internal)


class StreamMessage(BaseModel):
    """
    统一流式消息模型

    职责：抹平不同 Agent 框架的消息格式差异，统一系统内的消息流转
    """
    message_type: StreamMessageType
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    # 身份标识
    identity: Identity = Field(default_factory=Identity, description="身份标识")

    # 工具调用相关字段（可选）
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None

    @property
    def user_id(self) -> str:
        """获取用户 ID (兼容属性)"""
        return self.identity.user_id

    @property
    def agent_id(self) -> str:
        """获取 Agent ID (兼容属性)"""
        return self.identity.agent_id

    @property
    def session_id(self) -> Optional[str]:
        """获取会话 ID (兼容属性)"""
        return self.identity.session_id

    @property
    def role(self) -> str:
        """映射消息类型到 OpenAI 角色"""
        mapping = {
            StreamMessageType.USER: "user",
            StreamMessageType.ASSISTANT: "assistant",
            StreamMessageType.SYSTEM: "system",
            StreamMessageType.THOUGHT: "assistant",
            StreamMessageType.TOOL_CALL: "assistant",
            StreamMessageType.TOOL: "tool",
        }
        return mapping.get(self.message_type, "assistant")

    @property
    def token_count(self) -> int:
        """估算消息的 Token 数量"""
        return estimate_tokens(self.content)

    def to_langchain_message(self) -> Dict[str, str]:
        """转换为 LangChain 消息格式"""
        return {
            "role": self.role,
            "content": self.content
        }

    model_config = ConfigDict(use_enum_values=True)
