"""Agent 请求/响应模型

请求只约束形状：字段取值由 Patchouli 按读取 Profile 文件的同一套规则校验，不合格时
以 ``workspace.agent_profile.invalid`` 拒绝。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from hivememory.core.models import StoredAgentProfile


class AgentCreateRequest(BaseModel):
    """新建一个 Agent Profile；``agent_id`` 是文件名，创建后不可改。"""

    agent_id: str
    title: str = ""
    summary: str = ""
    persona: str = ""
    model: str = "default"
    temperature: float | None = None
    top_p: float | None = None
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")

    def fields(self) -> dict[str, object]:
        """除 ``agent_id`` 之外的全部字段，交给 Patchouli 校验。"""
        return self.model_dump(exclude={"agent_id"})


class AgentUpdateRequest(BaseModel):
    """修改一个 Agent Profile：只改请求中出现的字段，``null`` 的采样参数表示沿用模型默认值。

    没有 ``agent_id`` 字段：它不可改，出现在请求中按未知字段拒绝。
    """

    title: str | None = None
    summary: str | None = None
    persona: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")

    def changes(self) -> dict[str, object]:
        """请求中出现的字段；未出现的保持不变。"""
        return self.model_dump(exclude_unset=True)


class AgentProfileResponse(BaseModel):
    agent_id: str
    title: str
    summary: str
    persona: str
    model: str
    temperature: float | None
    top_p: float | None
    enabled: bool
    # 文件的修改时间：经 Agent 页保存与直接改文件都会推进它。
    updated_at: datetime

    @classmethod
    def from_stored(cls, stored: StoredAgentProfile) -> "AgentProfileResponse":
        profile = stored.profile
        return cls(
            agent_id=profile.agent_id or "",
            title=profile.title,
            summary=profile.summary,
            persona=profile.persona,
            model=profile.model,
            temperature=profile.temperature,
            top_p=profile.top_p,
            enabled=profile.enabled,
            updated_at=stored.saved_at,
        )
