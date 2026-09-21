"""访问控制配置：System 接入登记与 Workspace Actor 访问登记的本地声明。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 1/2.2 节）：
首版采用明确的本地配置和进程内不可变注册表，配置修改经重启生效；
System composition 负责装载本节配置并注入统一认证网关与共享行为检查。
缺省（空配置）即 fail closed——所有网关认证被拒绝；既有裸 scope 兼容
链路不受影响，其收紧由 A6 完成消费者切换后执行。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AccessControlConfig",
    "SystemPrincipalAccessEntry",
    "WorkspaceActorAccessEntry",
]


class SystemPrincipalAccessEntry(BaseModel):
    """System 接入登记配置：一个受信调用来源。

    ``adapters`` 声明可经哪些 adapter 接入（本地进程内调用方使用
    ``local``）；``allowed_user_ids`` 为可选身份解析收紧，``None`` 表示
    不按用户限定。principal 配置不授予任何 Workspace operation。
    """

    principal_id: str = Field(description="稳定调用来源标识（带命名空间）")
    kind: str = Field(default="local-process", description="连接/接收方式标识")
    enabled: bool = Field(default=True, description="是否启用该接入来源")
    adapters: list[str] = Field(
        default_factory=lambda: ["local"],
        description="该来源可使用的 adapter 标识",
    )
    allowed_user_ids: list[str] | None = Field(
        default=None,
        description="可选：该来源可服务的用户集合（身份解析收紧）",
    )

    # 访问控制配置拼错字段会被静默丢弃并落到 deny-by-default，方向虽安全
    # 但掩盖配置错误；本节选择 forbid 在装载期显式失败。
    model_config = ConfigDict(extra="forbid")


class WorkspaceActorAccessEntry(BaseModel):
    """Workspace Actor 访问登记配置：准入状态 + 行为白名单。

    ``allowed_operations`` 使用 ``WorkspaceOperation`` 的枚举值字符串
    （如 ``"resource.read"``）；允许为空——代表可进入但未获准执行资源
    操作。W0 兼容基线要求 ``user_id == owner_user_id``。
    """

    owner_user_id: str = Field(description="Workspace 归属 owner 用户 ID")
    workspace_id: str = Field(description="Workspace 标识")
    user_id: str = Field(description="Actor 用户 ID（W0 基线：等于 owner_user_id）")
    agent_id: str = Field(description="Actor Agent ID")
    enabled: bool = Field(default=True, description="是否允许进入该 Workspace")
    allowed_operations: list[str] = Field(
        default_factory=list,
        description="进入后允许执行的 operation 枚举值（行为上限，可为空）",
    )

    model_config = ConfigDict(extra="forbid")


class AccessControlConfig(BaseModel):
    """访问控制根配置：两类登记 + 认证有效区间声明。

    ``context_ttl_seconds`` 为 ``None`` 表示不设固定 TTL，context 仅随
    网关/运行实例生命周期失效（A1 第 3.4 节）。
    """

    principals: list[SystemPrincipalAccessEntry] = Field(
        default_factory=list,
        description="System Actor 接入登记",
    )
    workspace_actors: list[WorkspaceActorAccessEntry] = Field(
        default_factory=list,
        description="Workspace Actor 访问登记（准入 + 行为白名单）",
    )
    context_ttl_seconds: float | None = Field(
        default=None,
        description="context 认证有效区间上限（秒）；None 表示随网关生命周期",
    )

    model_config = ConfigDict(extra="forbid")
