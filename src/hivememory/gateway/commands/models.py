from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hivememory.core.models import FrozenDict, freeze_mapping


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


class CommandRouteTarget(BaseModel):
    """指令路由目标描述。"""

    kind: CommandRouteTargetKind = Field(default=CommandRouteTargetKind.LOCAL_HANDLER)
    name: str
    payload_template: FrozenDict[str, Any] = Field(default_factory=FrozenDict)

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("payload_template", mode="before")
    @classmethod
    def _freeze_payload(cls, value: Any) -> FrozenDict[str, Any]:
        return freeze_mapping(value)


class CommandPermissionPolicy(BaseModel):
    """指令可见性与执行权限策略。"""

    visibility: Literal["public", "debug", "admin"] = "public"
    allowed_agent_ids: tuple[str, ...] | None = None
    allowed_user_ids: tuple[str, ...] | None = None
    requires_confirmation: bool = False
    destructive: bool = False

    model_config = ConfigDict(frozen=True)


class CommandDefinition(BaseModel):
    """全局系统指令定义。"""

    command_id: str
    category: CommandCategory
    primary_name: str
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    summary: str
    description: str | None = None
    argument_schema: FrozenDict[str, Any] = Field(default_factory=FrozenDict)
    route_target: CommandRouteTarget
    permission: CommandPermissionPolicy = Field(default_factory=CommandPermissionPolicy)
    enabled: bool = True
    hidden: bool = False
    priority: int = 100

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("argument_schema", mode="before")
    @classmethod
    def _freeze_argument_schema(cls, value: Any) -> FrozenDict[str, Any]:
        return freeze_mapping(value)


class CommandParseResult(BaseModel):
    """系统指令解析产物，由 Gateway S0 透传给后续应用层。"""

    command_id: str | None = None
    raw_input: str
    name: str = ""
    args: FrozenDict[str, Any] = Field(default_factory=FrozenDict)
    tokens: tuple[str, ...] = Field(default_factory=tuple)
    matched_alias: str | None = None
    parse_status: CommandParseStatus
    error: str | None = None

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("args", mode="before")
    @classmethod
    def _freeze_args(cls, value: Any) -> FrozenDict[str, Any]:
        return freeze_mapping(value)


__all__ = [
    "CommandCategory",
    "CommandDefinition",
    "CommandParseResult",
    "CommandParseStatus",
    "CommandPermissionPolicy",
    "CommandRouteTarget",
    "CommandRouteTargetKind",
]
