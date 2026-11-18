"""手工管理记忆的公共契约：创建与编辑的材料，以及列表的一页结果。

调用方只提交材料；记忆 ID、时间戳、来源记录与生命周期状态都由 Patchouli
在提交边界确定，来源记录取自本次请求的发起者，没有给出读取策略时为 PUBLIC。字段的
取值约束由 Patchouli 的记忆模型校验，非法取值以 ``InvalidMemoryFieldError``
拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.models import MemoryAccessPolicy, MemoryAtom, MemoryType


@dataclass(frozen=True)
class MemoryCreation:
    """手工创建一条记忆的材料。``access_policy`` 为 ``None`` 时为 PUBLIC。"""

    title: str
    content: str
    memory_type: MemoryType
    summary: str = ""
    tags: tuple[str, ...] = ()
    alias: str | None = None
    access_policy: MemoryAccessPolicy | None = None


@dataclass(frozen=True)
class MemoryEdit:
    """对一条已有记忆的部分编辑。

    字段为 ``None`` 表示不修改；空白 alias 表示清除。全部为 ``None`` 或规范化后
    与当前值相同的编辑不产生新版本，原样返回当前原子。修改 ``access_policy``
    不产生新版本。
    """

    title: str | None = None
    summary: str | None = None
    content: str | None = None
    alias: str | None = None
    tags: tuple[str, ...] | None = None
    access_policy: MemoryAccessPolicy | None = None


@dataclass(frozen=True)
class MemoryListPage:
    """记忆列表的一页。

    没有检索词时按存储顺序翻页：把 ``next_cursor`` 交回下一次请求即取得下一页，
    为 ``None`` 表示已经取完；存储顺序与时间无关，需要按时间排列的调用方取全后
    自行排序。``total`` 是同一过滤条件下目标 Workspace 的记忆总数。

    有检索词时结果按相关度排列、不翻页：``next_cursor`` 为 ``None``，``total``
    是本次返回的条数。
    """

    memories: tuple[MemoryAtom, ...]
    next_cursor: str | None
    total: int


__all__ = ["MemoryCreation", "MemoryEdit", "MemoryListPage"]
