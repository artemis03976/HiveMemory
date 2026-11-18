"""
HiveMemory 核心数据模型 - 智能体领域

定义与多智能体系统（Agentic System）相关的数据模型。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentProfile(BaseModel):
    """
    人偶图纸 - Agent 运行时的完整配置信息

    一份 Profile 就是 AgentProfileStore 中的一个文件：文件名是 ``agent_id``，
    正文是 ``persona``，其余字段是文件的 frontmatter。内置的 omni_doll 不对应文件。

    三种名字各有用途：``title`` 是用户给它起的名字，用于展示；``summary`` 描述它
    负责的功能；``agent_id`` 用于系统内部的流转（访问登记、CALL 的目标、会话与
    记忆来源中的引用），创建后不可改。

    Agent 的操作权限只由 Workspace 访问登记表达，不附着在 Profile 上；Profile 也
    没有可见性，谁能调用它由调用方的 CALL 授权与它在 Workspace 中的准入决定。
    """

    agent_id: str | None = Field(
        default=None, description="Agent 身份，即 Profile 文件名；内置 Profile 为 omni_doll"
    )
    title: str = Field(default="", max_length=200, description="用户给它起的名字，用于展示")
    summary: str = Field(default="", max_length=500, description="它负责的功能，一句话")
    persona: str = Field(default="", description="Agent 人设提示词")
    model: str = Field(default="default", min_length=1, description="模型注册表中的模型 id")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="推理温度覆盖。None 表示沿用注册表模型定义的温度"
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="核采样阈值覆盖。None 表示沿用注册表模型定义的 top_p",
    )

    enabled: bool = Field(
        default=True,
        description="是否可以用于执行；停用的 Profile 在解析时被拒绝，也不出现在可用 agent 列表中",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "summary", "persona", "model", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        """首尾空白不构成内容；手改文件时编辑器留下的空行因此不影响往返一致。"""
        return value.strip() if isinstance(value, str) else value


class StoredAgentProfile(BaseModel):
    """AgentProfileStore 中的一份 Profile 与它的保存时间。

    ``saved_at`` 取文件的修改时间（UTC），供 Agent 页显示“最近保存”；直接改动
    文件同样会推进它。Profile 解析只交出 ``profile``，保存时间不进入运行。
    """

    profile: AgentProfile
    saved_at: datetime

    model_config = ConfigDict(frozen=True)


OMNI_DOLL_PROFILE = AgentProfile(
    agent_id="omni_doll",
    persona="",
    model="default",
    temperature=None,
    top_p=None,
)
"""全能人偶 (Omni-Doll) 默认配置，无特定人设。

temperature/top_p 为 None，运行时沿用注册表默认模型的参数；操作权限由访问登记提供。
"""
