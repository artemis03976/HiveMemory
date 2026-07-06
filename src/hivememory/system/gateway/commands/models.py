from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CommandCategory(str, Enum):
    """系统指令所属的能力域，用于 listing、权限和后续路由分组。"""

    SYSTEM = "system"
    DEBUG = "debug"
    PATCHOULI = "patchouli"
    RUNTIME = "runtime"


class CommandParseStatus(str, Enum):
    """L1 指令解析结果状态；这里只描述解析，不代表执行结果。"""

    MATCHED = "matched"
    INVALID_ARGS = "invalid_args"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class CommandRouteTargetKind(str, Enum):
    """指令命中后的目标类型，实际副作用由 Phase 2.2 dispatcher 执行。"""

    LOCAL_HANDLER = "local_handler"
    GLOBAL_ROUTE = "global_route"
    CLIENT_ACTION = "client_action"
    FUTURE_JOB = "future_job"


class CommandExecutionStatus(str, Enum):
    """统一指令执行结果状态；Phase 2.0/2.1 只定义协议，不执行指令。"""

    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    NOT_IMPLEMENTED = "not_implemented"


class CommandRouteTarget(BaseModel):
    """指令路由目标描述。"""

    kind: CommandRouteTargetKind = Field(default=CommandRouteTargetKind.LOCAL_HANDLER)
    name: str
    payload_template: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class CommandPermissionPolicy(BaseModel):
    """指令可见性与执行权限策略。"""

    visibility: Literal["public", "debug", "admin"] = "public"
    allowed_agent_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None
    requires_confirmation: bool = False
    destructive: bool = False


class CommandDefinition(BaseModel):
    """全局系统指令定义。"""

    command_id: str
    category: CommandCategory
    primary_name: str
    aliases: list[str] = Field(default_factory=list)
    summary: str
    description: str | None = None
    argument_schema: dict[str, Any] = Field(default_factory=dict)
    route_target: CommandRouteTarget
    permission: CommandPermissionPolicy = Field(default_factory=CommandPermissionPolicy)
    enabled: bool = True
    hidden: bool = False
    priority: int = 100

    model_config = ConfigDict(use_enum_values=True)


class CommandParseResult(BaseModel):
    """系统指令解析产物，由 RuleInterceptor 透传给 TheEye 和后续应用层。"""

    command_id: str | None = None
    raw_input: str
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    tokens: list[str] = Field(default_factory=list)
    matched_alias: str | None = None
    parse_status: CommandParseStatus
    error: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class CommandExecutionResult(BaseModel):
    """系统指令执行产物。当前阶段只定义协议，执行入口后续补齐。"""

    command_id: str
    status: CommandExecutionStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    client_action: dict[str, Any] | None = None
    error_code: str | None = None

    model_config = ConfigDict(use_enum_values=True)


__all__ = [
    "CommandCategory",
    "CommandDefinition",
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "CommandParseResult",
    "CommandParseStatus",
    "CommandPermissionPolicy",
    "CommandRouteTarget",
    "CommandRouteTargetKind",
]
