"""接入登记与 Workspace 访问记录的请求/响应模型。

凭据明文只出现在登记与轮换的响应中，供用户复制到 harness 一侧；列表与其他
响应只给打码值。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.access import WorkspaceOperation
from hivememory.server.models.secrets import mask_api_key
from hivememory.system.access import SystemActorAccessEntry
from hivememory.workspace.registry import WorkspaceActorAccessRecord

type ExternalAdapter = Literal["mcp", "import"]


class PrincipalResponse(BaseModel):
    """一条接入登记；进程内来源 ``editable`` 为 false，只能经登记文件修改。"""

    principal_id: str
    kind: str
    enabled: bool
    adapters: list[str]
    allowed_agent_ids: list[str] | None = Field(
        description="可声明的 Agent；null 表示不限定",
    )
    allow_system_actor: bool
    editable: bool = Field(description="是否可以经接口修改：只有外部来源可以")
    credential_source: Literal["env", "generated"] | None = Field(
        description="生效凭据的来源；外部来源为 null 时没有可用凭据，认证一律被拒绝",
    )
    credential_env: str | None = Field(description="登记的凭据环境变量名")
    credential_masked: str | None = Field(description="脱敏后的生效凭据")

    @classmethod
    def from_entry(cls, entry: SystemActorAccessEntry) -> "PrincipalResponse":
        return cls(
            principal_id=entry.principal_id,
            kind=entry.kind,
            enabled=entry.enabled,
            adapters=sorted(entry.adapters),
            allowed_agent_ids=(
                sorted(entry.allowed_agent_ids) if entry.allowed_agent_ids is not None else None
            ),
            allow_system_actor=entry.allow_system_actor,
            editable=not entry.is_in_process,
            credential_source=entry.credential_source,
            credential_env=entry.credential_env,
            credential_masked=mask_api_key(entry.credential),
        )


class PrincipalCredentialResponse(BaseModel):
    """登记或轮换的结果：凭据明文只在这里返回一次。"""

    principal: PrincipalResponse
    credential: str = Field(description="新凭据的明文，写进 harness 一侧的环境变量")


class PrincipalSettings(BaseModel):
    """外部来源可以修改的内容。"""

    adapters: list[ExternalAdapter] = Field(min_length=1, description="外部 adapter 的非空子集")
    allowed_agent_ids: list[str] | None = Field(
        default=None,
        description="可声明的 Agent；null 表示不限定",
    )
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _agents_not_blank(self) -> "PrincipalSettings":
        if self.allowed_agent_ids is not None and any(
            not agent_id.strip() for agent_id in self.allowed_agent_ids
        ):
            raise ValueError("可声明的 Agent 不能为空字符串")
        return self


class PrincipalCreateRequest(PrincipalSettings):
    """登记一个外部来源。"""

    principal_id: str = Field(min_length=1, description="稳定来源标识，例如 harness:my-tool")

    @model_validator(mode="after")
    def _principal_not_blank(self) -> "PrincipalCreateRequest":
        if not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
        return self


class ActorRecordResponse(BaseModel):
    """一条 Workspace 访问记录；``agent_id`` 为 null 表示用户级记录。"""

    agent_id: str | None = Field(description="null 表示用户级记录，覆盖其余所有具体 Agent")
    enabled: bool
    allowed_operations: list[WorkspaceOperation]

    @classmethod
    def from_record(cls, record: WorkspaceActorAccessRecord) -> "ActorRecordResponse":
        order = list(WorkspaceOperation)
        return cls(
            agent_id=record.agent_id,
            enabled=record.enabled,
            allowed_operations=sorted(record.allowed_operations, key=order.index),
        )


class ActorRecordRequest(BaseModel):
    """新增或修改一条访问记录；禁用的记录不能带白名单。"""

    enabled: bool = True
    allowed_operations: list[WorkspaceOperation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _disabled_without_operations(self) -> "ActorRecordRequest":
        if not self.enabled and self.allowed_operations:
            raise ValueError("禁用的访问记录不能配置操作白名单")
        return self
