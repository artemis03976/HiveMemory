"""Chat 相关的 Request/Response 模型"""


from pydantic import BaseModel, Field

from hivememory.core.constants import DEFAULT_AGENT_ID, DEFAULT_USER_ID


class GenerationOptions(BaseModel):
    model: str | None = Field(default=None, min_length=1, description="模型名称")
    temperature: float | None = Field(default=None, ge=0, le=2, description="采样温度")
    top_p: float | None = Field(default=None, ge=0, le=1, description="Top-p 采样率")
    max_tokens: int | None = Field(default=None, ge=1, le=32768, description="最大生成长度")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    session_id: str | None = Field(default=None, description="会话 ID")
    enable_memory_retrieval: bool = Field(default=True, description="是否启用记忆检索")
    generation_options: GenerationOptions | None = Field(default=None, description="本次请求的生成参数覆盖")


class StopChatRequest(BaseModel):
    generation_id: str = Field(..., description="要停止的生成任务 ID")


# ========== SSE 事件数据模型 ==========

class StreamNamespace(BaseModel):
    """流式事件命名空间：用于区分主/子 Agent 输出来源。"""

    scope: str | None = Field(default=None, description="事件作用域：main 或 sub")
    depth: int | None = Field(default=None, description="当前执行深度，主帧通常为 0")
    agent_id: str | None = Field(default=None, description="当前输出的 agent 标识")
    frame_id: str | None = Field(default=None, description="当前执行帧 ID")


class ChatTokenEvent(StreamNamespace):
    """token 事件: LLM 生成的文本增量"""
    content: str


class MTPStartEvent(StreamNamespace):
    """mtp_start 事件: MTP 指令被拦截"""
    verb: str
    target: str = ""
    args: dict = Field(default_factory=dict)
    raw_text: str = ""
    iteration: int


class MTPResultEvent(StreamNamespace):
    """mtp_result 事件: MTP 执行完成"""
    verb: str
    target: str = ""
    args: dict = Field(default_factory=dict)
    raw_text: str = ""
    status: str
    iteration: int


class SubAgentStartEvent(StreamNamespace):
    """sub_agent_start 事件: 子 Agent 生命周期开始。"""

    agent_id: str
    task: str
    iteration: int


class SubAgentEndEvent(StreamNamespace):
    """sub_agent_end 事件: 子 Agent 生命周期结束。"""

    status: str
    final_text: str | None = None
    iteration: int


class TopicInfoEvent(BaseModel):
    """topic_info 事件: 话题路由结果"""
    topic_id: str
    is_new: bool
    pool_topics: list[dict] = Field(default_factory=list)


class ChatDoneEvent(BaseModel):
    """done 事件: 生成完成"""
    final_text: str
    mtp_iterations: int
    total_iterations: int
    generation_id: str | None = None
    status: str = "completed"
    stopped: bool = False
    reason: str | None = None
    memory_task_ids: list[str] = Field(default_factory=list)
    pool_topics: list[dict] = Field(default_factory=list)


class CommandResultEvent(BaseModel):
    """command_result 事件：Gateway 系统指令执行结果。"""

    command_id: str
    status: str
    message: str
    data: dict = Field(default_factory=dict)
    client_action: dict | None = None
    error_code: str | None = None


class ChatErrorEvent(BaseModel):
    """error 事件: 错误发生"""
    message: str
    detail: str | None = None
