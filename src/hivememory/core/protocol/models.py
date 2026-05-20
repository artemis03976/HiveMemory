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

from pydantic import BaseModel, Field, ConfigDict

from hivememory.engines.retrieval.models import QueryFilters
from hivememory.core.models import AgentProfile, MemoryAtom, Identity, TraceItem, TurnEvent
from hivememory.engines.gateway.models import GatewayIntent

# QueryFilters 的规范定义位于引擎层，此处重导出以保持向后兼容
from hivememory.engines.retrieval.models import QueryFilters


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


class RetrievalRequest(ProtocolMessage):
    """
    检索请求协议消息

    从 Eye 发送到 RetrievalFamiliar 的检索请求。
    用于热路径 (Hot Path) 的实时记忆检索。

    乐观检索策略：
    - 基础过滤条件由 RetrievalFamiliar 根据 identity 动态创建
    - MTP SEARCH 指令可通过 filters 字段叠加额外过滤维度 (如 type:CODE)

    Attributes:
        msg_type: 固定为 RETRIEVAL_REQUEST
        semantic_query: 指代消解后的完整查询，用于语义检索
        keywords: 稀疏检索关键词列表（BM25）
        identity: 请求者身份标识 (user_id + agent_id + team_id)
        filters: MTP filter 解析后的过滤条件 (可选，叠加到 identity 基线之上)

    Examples:
        >>> request = RetrievalRequest(
        ...     semantic_query="如何部署贪吃蛇游戏？",
        ...     keywords=["部署", "贪吃蛇", "游戏"],
        ...     identity=Identity(user_id="user123")
        ... )
    """

    msg_type: MessageType = MessageType.RETRIEVAL_REQUEST

    # 指代消解后的完整查询（用于语义检索）
    semantic_query: str = Field(..., description="指代消解后的查询")

    # 稀疏检索关键词（用于 BM25）
    keywords: List[str] = Field(default_factory=list, description="检索关键词")

    # 请求者身份标识
    identity: Identity = Field(default_factory=Identity, description="请求者身份标识")

    # MTP SEARCH 指令传入的过滤条件 (可选)
    filters: Optional[QueryFilters] = Field(default=None, description="MTP filter 过滤条件")

    @property
    def user_id(self) -> str:
        """兼容属性: 从 identity 中提取 user_id"""
        return self.identity.user_id


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


class AgentRunContext(BaseModel):
    """Neutral context prepared for Alice to assemble and execute an Agent run."""

    identity: Identity = Field(default_factory=Identity)
    topic_id: str = Field(default="")
    user_message: str = Field(default="")
    topic_context: Dict[str, Any] = Field(default_factory=dict)
    retrieval_result: RetrievalResponse = Field(default_factory=RetrievalResponse)
    agent_profile: AgentProfile
    storage_available: bool = Field(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    主动 chat 主链路的统一返回值

    封装 Kernel 递归生成循环的完整结果，包含最终文本和 MTP 执行统计。

    Attributes:
        final_text: 用户可见的最终回复文本 (仅自然语言部分，不含 MTP 指令/XML)
        mtp_iterations: MTP 中断执行次数
        total_iterations: 总生成轮次 (含最终的非 MTP 轮)
    """
    final_text: str = Field(default="", description="用户可见的最终回复文本")
    mtp_iterations: int = Field(default=0, description="MTP 中断次数")
    total_iterations: int = Field(default=1, description="总生成轮次")
    # LoopExecutor 收集的结构化轮次事件（序列化为 dict 避免循环导入）
    turn_events: List[Any] = Field(default_factory=list, description="LoopExecutor 收集的 TurnEvent 列表")


class EyeGazeResult(BaseModel):
    """
    TheEye 的统一输出模型

    TheEye 作为 Agentic Dispatcher，负责信息重整与话题路由。
    数据格式转换由 PatchouliRuntime 负责。

    Attributes:
        intent: Gateway 意图分类
        rewritten_query: 指代消解后的完整查询
        search_keywords: 稀疏检索关键词列表
        worth_saving: 是否值得保存
        raw_query: 原始用户查询
        identity: 身份标识
        processing_time_ms: Eye 处理耗时（毫秒）
        is_fallback: 是否为 fallback 结果
        target_topic: 路由目标话题 ID 或 "NEW_TOPIC"
    """
    intent: GatewayIntent = Field(..., description="意图分类")
    rewritten_query: str = Field(..., description="指代消解后的查询")
    search_keywords: List[str] = Field(default_factory=list, description="检索关键词")
    worth_saving: bool = Field(..., description="是否值得保存")
    raw_query: str = Field(..., description="原始用户查询")
    identity: Identity = Field(default_factory=Identity, description="身份标识")
    processing_time_ms: float = Field(default=0.0, description="处理耗时")
    is_fallback: bool = Field(default=False, description="是否为 fallback 结果")

    #: 路由目标话题 (MMU Agentic Routing, Phase 4.5)
    target_topic: str = Field(default="NEW_TOPIC", description="路由目标话题 ID 或 NEW_TOPIC")

    #: 新话题标题（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_title: Optional[str] = Field(default=None, description="新话题标题")

    #: 新话题摘要（仅 NEW_TOPIC 时由 Gateway 生成）
    new_topic_summary: Optional[str] = Field(default=None, description="新话题摘要")


class InteractionPayload(BaseModel):
    """
    PatchouliSystem / Kernel -> Perception 的原子交互传输包

    作为系统级数据传输协议存在，承载单轮交互在进入感知层前的完整结构化结果。

    Attributes:
        user_message: 原始用户消息
        mtp_traces: Koakuma 执行期记录的 Trace 列表
        write_focus: WRITE 指令核心素材
        update_focus: UPDATE 指令修改意图
        identity: 归属身份元数据
        rewritten_query: Gateway 重写后的查询
        worth_saving: Gateway 价值判断
        assistant_final_text: 最终自然语言回复
        turn_events: 结构化轮次事件列表
    """
    # 身份元数据
    identity: Identity = Field(default_factory=Identity, description="归属元数据")

    user_message: str = Field(..., description="原始用户消息")

    rewritten_query: Optional[str] = Field(
        default=None,
        description="Gateway 重写后的查询"
    )

    # ========== 结构化轮次事件 ==========
    # 模型最终自然语言回复
    assistant_final_text: Optional[str] = Field(
        default=None,
        description="去除 MTP 噪音后的最终自然语言回复（loop_result.final_text 直传）"
    )
    # 收集的结构化轮次事件列表
    turn_events: List[TurnEvent] = Field(
        default_factory=list,
        description="LoopExecutor 收集的结构化轮次事件列表，有值时感知层优先走结构化路径"
    )
    mtp_traces: List[TraceItem] = Field(
        default_factory=list,
        description="由 Koakuma 在执行过程中记录的 Trace 列表"
    )

    # 控制信号 (挂载在 Payload 上，而非独立传输)
    write_focus: Optional[Any] = Field(
        default=None,
        description="WRITE 指令的核心素材 (WriteFocus)"
    )
    update_focus: Optional[Any] = Field(
        default=None,
        description="UPDATE 指令的修改意图 (UpdateFocus)"
    )

    worth_saving: Optional[bool] = Field(
        default=None,
        description="Gateway 价值判断"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnalyzeAndRetrieveResult(BaseModel):
    """Patchouli 标准分析与预检索组合结果。"""

    gaze_result: EyeGazeResult = Field(..., description="入口分析结果")
    retrieval_result: RetrievalResponse = Field(..., description="热路径预检索结果")


__all__ = [
    "MessageType",
    "ProtocolMessage",
    "QueryFilters",
    "RetrievalRequest",
    "RetrievalResponse",
    "AgentRunContext",
    "EyeGazeResult",
    "InteractionPayload",
    "AnalyzeAndRetrieveResult",
    "MTPExecutionResult",
]
