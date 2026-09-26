"""workspace memory read 能力：alias resolver 与 Profile 读取 resolver（A2 §2）。

resolver 在交付边界对当前 Actor 逐次授权，L2 冷读经 ``CanonicalReadBackend``
协议访问 Patchouli backing。本包只依赖 core 与 workspace 自身。
"""

from hivememory.workspace.resolution.alias import AliasResolver
from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard
from hivememory.workspace.resolution.profile import ProfileResolver

__all__ = [
    "AliasResolver",
    "CanonicalReadBackend",
    "ColdReadGuard",
    "ProfileResolver",
]
