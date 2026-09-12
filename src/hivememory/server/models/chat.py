"""Chat 相关的 Request/Response 模型"""

from pydantic import BaseModel, Field, model_validator

from hivememory.core.models import AttachmentSelectionRequest


class GenerationOptions(BaseModel):
    model: str | None = Field(default=None, min_length=1, description="模型名称")
    temperature: float | None = Field(default=None, ge=0, le=2, description="采样温度")
    top_p: float | None = Field(default=None, ge=0, le=1, description="Top-p 采样率")
    max_tokens: int | None = Field(default=None, ge=1, le=32768, description="最大生成长度")


class ChatRequest(BaseModel):
    """Chat 请求体。

    Chat 是 Agent action：``agent_id`` 必填且必须为具体 Agent，不得回退到
    保留 ``system`` actor。用户导向基础选择（``user_id + workspace_id``）
    一律由统一请求头（``x-user-id`` / ``x-workspace-id``）承载，body 不再
    重复传递，避免同一请求出现两种身份事实。

    ``attachments`` 是用户显式选择的附件数组（计划 9.2 节）：空数组与缺少
    字段等价；同一 ``asset_ref`` 只能出现一次，重复即请求冲突；顺序属于
    用户输入，保留进请求 canonical 表示。ref 归属与版本校验在 Patchouli
    prepare 边界完成，HTTP 层只校验字段类型与数组结构。
    """

    message: str = Field(..., description="用户消息")
    agent_id: str = Field(..., description="执行本次对话的具体 Agent ID")
    session_id: str | None = Field(default=None, description="会话 ID")
    enable_memory_retrieval: bool = Field(default=True, description="是否启用记忆检索")
    generation_options: GenerationOptions | None = Field(
        default=None, description="本次请求的生成参数覆盖"
    )
    attachments: list[AttachmentSelectionRequest] = Field(
        default_factory=list,
        description="本轮显式选择的附件坐标数组；空数组表示不使用附件",
    )

    @model_validator(mode="after")
    def _reject_duplicate_asset_refs(self) -> "ChatRequest":
        seen_tokens: set[str] = set()
        for selection in self.attachments:
            if selection.asset_ref.token in seen_tokens:
                raise ValueError("同一 asset_ref 在 attachments 中重复出现")
            seen_tokens.add(selection.asset_ref.token)
        return self


class StopChatRequest(BaseModel):
    """Stop 请求体。

    取消不是 Agent action：不携带 agent_id，取消与事件发布使用 generation
    创建时冻结在 registry 里的原始 scope。基础身份选择同样只来自统一请求头，
    由 server 用于 owner/workspace 校验。
    """

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
