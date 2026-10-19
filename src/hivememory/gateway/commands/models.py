from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hivememory.core.models import FrozenDict, freeze_mapping


class CommandCategory(str, Enum):
    """系统指令所属的能力域，用于 listing 分组。"""

    SYSTEM = "system"
    DEBUG = "debug"
    PATCHOULI = "patchouli"
    RUNTIME = "runtime"


class CommandDefinition(BaseModel):
    """全局系统指令定义：名称、别名、参数 schema 与展示信息。

    命令只解析不执行，定义不声明运行位置与执行权限；命令系统接回时随运行
    位置一并设计。
    """

    command_id: str
    category: CommandCategory
    primary_name: str
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    summary: str
    description: str | None = None
    argument_schema: FrozenDict[str, Any] = Field(default_factory=FrozenDict)
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


__all__ = [
    "CommandCategory",
    "CommandDefinition",
]
