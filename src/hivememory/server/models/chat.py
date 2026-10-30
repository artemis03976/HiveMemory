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

    ``attachments`` 是用户显式选择的附件数组：空数组与缺少
    字段等价；同一 ``asset_ref`` 只能出现一次，重复即请求冲突；顺序属于
    用户输入，保留进请求 canonical 表示。ref 归属与版本校验在任务进程的
    CPU 分配中随附件租借完成，HTTP 层只校验字段类型与数组结构。
    """

    # 用户消息是交互记录序号 0 的事件，每一轮都以它开头，因此不接受空消息。
    message: str = Field(..., min_length=1, description="用户消息，不能为空")
    agent_id: str = Field(..., description="执行本次对话的具体 Agent ID")
    session_id: str | None = Field(
        default=None, description="会话 ID；缺少时新建会话，不参与身份授权"
    )
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

    取消不是 Agent action：不携带 agent_id。基础身份选择同样只来自统一
    请求头，认证后得到请求级访问 context；进程控制授权以它比对进程记录
    持有的访问 context 的驻留坐标，不匹配与进程不存在一样返回
    ``not_found``。找到进程时，取消事件使用进程注册时绑定的观测标签。
    """

    process_id: str = Field(..., description="要停止的任务进程 ID")
