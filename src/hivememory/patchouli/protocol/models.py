"""
模块间通信协议模型

定义 Eye 与下游模块 (RetrievalFamiliar, LibrarianCore) 之间的通信协议。

作者: HiveMemory Team
版本: 2.0
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from hivememory.core.models import MemoryAtom, MemoryType, Identity

__all__ = [
    "MessageType",
    "ProtocolMessage",
    "QueryFilters",
    "RetrievalRequest",
    "Observation",
    "RetrievalResult",
]


class MessageType(str, Enum):
    """
    消息类型枚举

    定义模块间通信的消息类型，用于类型检查和路由。
    """

    # 检索请求 - Eye -> RetrievalFamiliar (热路径)
    RETRIEVAL_REQUEST = "retrieval_request"

    # 感知信号 - Eye -> LibrarianCore (冷路径)
    OBSERVATION = "observation"

    # 检索结果 - RetrievalFamiliar -> 外部Worker
    RETRIEVAL_RESULT = "retrieval_result"


class ProtocolMessage(BaseModel):
    """
    协议消息基类

    所有模块间通信消息的统一封装，提供：
    - 消息类型标识
    - 唯一消息 ID
    - 时间戳
    - 可扩展的上下文

    Attributes:
        msg_type: 消息类型
        msg_id: 唯一消息标识符
        timestamp: 消息创建时间
    """

    # 消息类型
    msg_type: MessageType

    # 唯一消息标识符（自动生成）
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # 消息创建时间
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryFilters(BaseModel):
    """
    结构化过滤条件协议模型

    用于 Gateway 和 RetrievalFamiliar 之间的过滤条件传递。
    这是共享协议的一部分，确保前后字段定义一致。

    Attributes:
        memory_type: 记忆类型过滤
        time_range: 时间范围过滤
        tags: 标签过滤
        source_agent_id: 来源 Agent ID 过滤
        user_id: 用户 ID 过滤
        min_confidence: 最小置信度过滤
    """
    memory_type: Optional[MemoryType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    tags: List[str] = Field(default_factory=list)
    source_agent_id: Optional[str] = None
    user_id: Optional[str] = None
    min_confidence: float = 0.0

    def is_empty(self) -> bool:
        """检查过滤条件是否为空"""
        return (
            self.memory_type is None
            and self.time_range is None
            and len(self.tags) == 0
            and self.source_agent_id is None
            and self.user_id is None
            and self.min_confidence == 0.0
        )


class RetrievalRequest(ProtocolMessage):
    """
    检索请求协议消息

    从 Eye 发送到 RetrievalFamiliar 的检索请求。
    用于热路径 (Hot Path) 的实时记忆检索。

    Attributes:
        msg_type: 固定为 RETRIEVAL_REQUEST
        semantic_query: 指代消解后的完整查询，用于语义检索
        keywords: 稀疏检索关键词列表（BM25）
        filters: 元数据过滤条件
        user_id: 用户标识符

    Examples:
        >>> from hivememory.core.models import MemoryType
        >>> request = RetrievalRequest(
        ...     semantic_query="如何部署贪吃蛇游戏？",
        ...     keywords=["部署", "贪吃蛇", "游戏"],
        ...     filters=QueryFilters(memory_type=MemoryType.FACT),
        ...     user_id="user123"
        ... )
    """

    msg_type: MessageType = MessageType.RETRIEVAL_REQUEST

    # 指代消解后的完整查询（用于语义检索）
    semantic_query: str = Field(..., description="指代消解后的查询")

    # 稀疏检索关键词（用于 BM25）
    keywords: List[str] = Field(default_factory=list, description="检索关键词")

    # 元数据过滤条件
    filters: QueryFilters = Field(default_factory=QueryFilters, description="过滤条件")

    # 用户标识符
    user_id: str = Field(default="default", description="用户 ID")


class Observation(ProtocolMessage):
    """
    感知信号协议消息

    从 Eye 发送到 LibrarianCore 的感知信号。
    用于冷路径 (Cold Path) 的记忆收集和处理。
    这是连接热路径和冷路径的关键消息。

    Attributes:
        msg_type: 固定为 OBSERVATION
        anchor: 语义锚点（Gateway 重写后的查询）
        raw_message: 原始用户消息
        role: 消息角色（user/assistant/system）
        identity: 对话ID标识（包含用户ID、AgentID、会话ID）
        gateway_context: Gateway 解析的元数据

    Examples:
        >>> observation = Observation(
        ...     anchor="如何部署贪吃蛇游戏？",
        ...     raw_message="怎么部署它？",
        ...     user_id="user123",
        ...     agent_id="chatbot",
        ...     gateway_context={
        ...         "intent": "RAG",
        ...         "worth_saving": True
        ...     }
        ... )
    """

    msg_type: MessageType = MessageType.OBSERVATION

    # 语义锚点（Gateway 重写后的查询）
    # 对于 Assistant/System 消息，此字段可能为空
    anchor: Optional[str] = Field(default=None, description="语义锚点")

    # 原始消息
    raw_message: str = Field(..., description="原始消息")

    # 消息角色
    role: str = Field(default="user", description="角色 (user/assistant/system)")

    # 标识符
    identity: Identity = Field(default_factory=Identity, description="对话ID标识")

    # Gateway 元数据
    gateway_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gateway 元数据",
    )


class RetrievalResult(ProtocolMessage):
    """
    检索结果协议消息
    
    从 RetrievalFamiliar 返回的检索结果，供外部 Worker Agent 使用
    包含完整的检索信息和渲染后的上下文
    """
    msg_type: MessageType = MessageType.RETRIEVAL_RESULT

    # 检索到的记忆
    memories: List[MemoryAtom] = Field(default_factory=list)  

    # 渲染后的上下文字符串
    rendered_context: str = ""  
    
    # 元信息
    latency_ms: float = 0.0  # 总耗时
    memories_count: int = 0  # 检索到的数量

    
    def is_empty(self) -> bool:
        """检查是否没有检索到任何记忆"""
        return len(self.memories) == 0
    
    def get_context_for_prompt(self) -> str:
        """获取可直接注入 System Prompt 的上下文"""
        if self.is_empty():
            return ""
        return self.rendered_context

