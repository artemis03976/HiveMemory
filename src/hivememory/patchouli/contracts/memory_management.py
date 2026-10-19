"""手工管理记忆与 Agent Profile 的公共输入契约：创建与编辑的材料。

调用方只提交材料；记忆 ID、时间戳、来源记录、默认访问策略与生命周期
状态都由 Patchouli 在提交边界确定，来源记录取自本次请求的发起者。字段的
取值约束由 Patchouli 的记忆模型校验，非法取值以 ``InvalidMemoryFieldError``
拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hivememory.core.models import MemoryType


@dataclass(frozen=True)
class MemoryCreation:
    """手工创建一条记忆的材料。

    ``memory_type`` 不能是 ``AGENT_PROFILE``：Profile 只经 Profile 创建入口创建，
    记忆创建入口按领域错误拒绝该类型。
    """

    title: str
    content: str
    memory_type: MemoryType
    summary: str = ""
    tags: tuple[str, ...] = ()
    alias: str | None = None


@dataclass(frozen=True)
class AgentProfileCreation:
    """创建一个 Agent Profile 的材料；Profile 按 alias 寻址，alias 必填。"""

    title: str
    alias: str
    summary: str = ""
    content: str = ""
    tags: tuple[str, ...] = ()
    agent_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryEdit:
    """对一条已有记忆的部分编辑；记忆与 Profile 共用。

    字段为 ``None`` 表示不修改；空白 alias 表示清除（普通记忆可以没有 alias，
    Profile 清除 alias 会被拒绝）。全部为 ``None`` 或规范化后与当前值相同的
    编辑不产生新版本，原样返回当前原子。
    """

    title: str | None = None
    summary: str | None = None
    content: str | None = None
    alias: str | None = None
    tags: tuple[str, ...] | None = None
    agent_config: dict[str, Any] | None = None


__all__ = ["AgentProfileCreation", "MemoryCreation", "MemoryEdit"]
