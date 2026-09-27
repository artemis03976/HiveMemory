"""Memory 授权谓词的兼容转发（A2 §8.2：实现已上收 ``hivememory.core.memory_access``）。

新代码直接从 core 导入；本模块只为既有引用保留转发，不维护第二份实现。
"""

from __future__ import annotations

from hivememory.core.memory_access import (
    access_policy_permits,
    memory_belongs_to_workspace,
    memory_is_readable,
    memory_visible_to_actor,
)

__all__ = [
    "access_policy_permits",
    "memory_belongs_to_workspace",
    "memory_visible_to_actor",
    "memory_is_readable",
]
