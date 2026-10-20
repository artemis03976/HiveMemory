"""workspace memory read 能力：记忆读取器与 Profile 读取器。

resolver 在交付边界对当前 Actor 逐次授权，L2 冷读经 ``CanonicalReadBackend``
协议访问 Patchouli backing。本包只依赖 core 与 workspace 自身。
"""

from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard
from hivememory.workspace.resolution.memory import MemoryReader
from hivememory.workspace.resolution.profile import ProfileResolver

__all__ = [
    "MemoryReader",
    "CanonicalReadBackend",
    "ColdReadGuard",
    "ProfileResolver",
]
