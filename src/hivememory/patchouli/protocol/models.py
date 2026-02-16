"""
模块间通信协议模型

定义 Eye 与 Kernel 之间，以及 Kernel 内部微服务之间的通信协议。

作者: HiveMemory Team
版本: 3.0
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hivememory.engines.retrieval.models import QueryFilters
from hivememory.core.models import MemoryAtom, Identity
from hivememory.engines.gateway.models import GatewayIntent

# QueryFilters 的规范定义位于引擎层，此处重导出以保持向后兼容
from hivememory.engines.retrieval.models import QueryFilters

class EyeGazeResult(BaseModel):
    """
    TheEye 的统一输出模型

    TheEye 作为 Ingress Gateway，只负责感知和信息重整，
    不构建下游协议消息（RetrievalRequest / Observation）。
    数据格式转换由 PatchouliKernel 负责。

    Attributes:
        intent: Gateway 意图分类
        rewritten_query: 指代消解后的完整查询
        search_keywords: 稀疏检索关键词列表
        worth_saving: 是否值得保存
        raw_query: 原始用户查询
        identity: 身份标识
        processing_time_ms: Eye 处理耗时（毫秒）
        is_fallback: 是否为 fallback 结果
    """
    intent: GatewayIntent = Field(..., description="意图分类")
    rewritten_query: str = Field(..., description="指代消解后的查询")
    search_keywords: List[str] = Field(default_factory=list, description="检索关键词")
    worth_saving: bool = Field(..., description="是否值得保存")
    raw_query: str = Field(..., description="原始用户查询")
    identity: Identity = Field(default_factory=Identity, description="身份标识")
    processing_time_ms: float = Field(default=0.0, description="处理耗时")
    is_fallback: bool = Field(default=False, description="是否为 fallback 结果")


class KernelHotResult(BaseModel):
    """
    PatchouliKernel 热路径的统一输出模型

    替代 handle_hot() 返回的 bare Dict[str, Any]，提供类型安全的返回值。

    Attributes:
        intent: 意图分类字符串
        rewritten: 重写后的查询
        keywords: 关键词列表
        worth_saving: 是否值得保存
        memory: 检索到的记忆上下文（可能为 None）
    """
    intent: str = Field(..., description="意图")
    rewritten: Optional[str] = Field(default=None, description="重写后的查询")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    worth_saving: bool = Field(default=False, description="是否值得保存")
    memory: Optional[str] = Field(default=None, description="检索到的记忆上下文")


__all__ = [
    "MessageType",
    "ProtocolMessage",
    "QueryFilters",
    "RetrievalRequest",
    "Observation",
    "RetrievalResponse",
    "EyeGazeResult",
    "KernelHotResult",
    "MTPExecutionResult",
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
    RETRIEVAL_RESPONSE = "retrieval_response"

    # MTP 指令与响应（Koakuma 服务）
    MTP_COMMAND = "mtp_command"
    MTP_RESPONSE = "mtp_response"


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


class Observation(ProtocolMessage):
    """
    感知信号协议消息

    从 Kernel 发送到 LibrarianCore 的感知信号。
    用于冷路径 (Cold Path) 的记忆收集和处理。

    Attributes:
        msg_type: 固定为 OBSERVATION
        anchor: 语义锚点（Gateway 重写后的查询）
        raw_message: 原始用户消息
        role: 消息角色（user/assistant/system）
        identity: 对话ID标识（包含用户ID、AgentID、会话ID）
        worth_saving: 是否值得保存为长期记忆

    Examples:
        >>> observation = Observation(
        ...     anchor="如何部署贪吃蛇游戏？",
        ...     raw_message="怎么部署它？",
        ...     worth_saving=True,
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

    # 是否值得保存为长期记忆
    worth_saving: bool = Field(default=False, description="是否值得保存")


class RetrievalRequest(ProtocolMessage):
    """
    检索请求协议消息

    从 Eye 发送到 RetrievalFamiliar 的检索请求。
    用于热路径 (Hot Path) 的实时记忆检索。

    乐观检索策略：
    - 基础过滤条件由 RetrievalFamiliar 根据 user_id 动态创建
    - MTP SEARCH 指令可通过 filters 字段叠加额外过滤维度 (如 type:CODE)

    Attributes:
        msg_type: 固定为 RETRIEVAL_REQUEST
        semantic_query: 指代消解后的完整查询，用于语义检索
        keywords: 稀疏检索关键词列表（BM25）
        user_id: 用户标识符
        filters: MTP filter 解析后的过滤条件 (可选，叠加到 user_id 基线之上)

    Examples:
        >>> request = RetrievalRequest(
        ...     semantic_query="如何部署贪吃蛇游戏？",
        ...     keywords=["部署", "贪吃蛇", "游戏"],
        ...     user_id="user123"
        ... )
    """

    msg_type: MessageType = MessageType.RETRIEVAL_REQUEST

    # 指代消解后的完整查询（用于语义检索）
    semantic_query: str = Field(..., description="指代消解后的查询")

    # 稀疏检索关键词（用于 BM25）
    keywords: List[str] = Field(default_factory=list, description="检索关键词")

    # 用户标识符
    user_id: str = Field(default="default", description="用户 ID")

    # MTP SEARCH 指令传入的过滤条件 (可选)
    filters: Optional[QueryFilters] = Field(default=None, description="MTP filter 过滤条件")


class RetrievalResponse(ProtocolMessage):
    """
    检索结果协议消息
    
    从 RetrievalFamiliar 返回的检索结果，供外部 Worker Agent 使用
    包含完整的检索信息和渲染后的上下文
    """
    msg_type: MessageType = MessageType.RETRIEVAL_RESPONSE

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


class MTPExecutionResult(BaseModel):
    """
    MTP 指令执行结果

    Kernel 级别的 MTP 执行结果封装，由 KoakumaRuntime 返回。
    包含解析后的指令、执行响应和格式化后的回填文本。

    Attributes:
        command: 解析后的 MTP 指令 (解析失败时为 None)
        response_status: 响应状态 (success/error/ack)
        response_content: 响应内容
        formatted_response: 格式化后的完整回填文本 (指令 + XML 响应容器)
        success: 是否执行成功
        execution_time_ms: 执行耗时 (毫秒)
    """
    command: Optional[Any] = Field(default=None, description="解析后的 MTPCommand 对象")
    response_status: str = Field(default="error", description="响应状态")
    response_content: str = Field(default="", description="响应内容")
    formatted_response: str = Field(default="", description="格式化后的回填文本")
    success: bool = Field(default=False, description="是否执行成功")
    execution_time_ms: float = Field(default=0.0, description="执行耗时 (毫秒)")


class ChatResult(BaseModel):
    """
    PatchouliSystem.chat() 的返回值

    封装 Kernel 递归生成循环的完整结果，包含最终文本和 MTP 执行统计。

    Attributes:
        final_text: 用户可见的最终回复文本 (仅自然语言部分，不含 MTP 指令/XML)
        mtp_iterations: MTP 中断执行次数
        total_iterations: 总生成轮次 (含最终的非 MTP 轮)
        mtp_commands_executed: 执行过的 MTP 指令动词列表 (如 ["SEARCH", "READ"])
    """
    final_text: str = Field(default="", description="用户可见的最终回复文本")
    mtp_iterations: int = Field(default=0, description="MTP 中断次数")
    total_iterations: int = Field(default=1, description="总生成轮次")
    mtp_commands_executed: List[str] = Field(default_factory=list, description="执行过的 MTP 指令摘要")
