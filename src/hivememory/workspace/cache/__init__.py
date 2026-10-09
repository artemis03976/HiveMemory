"""Workspace 派生缓存：完整原子缓存、Profile 解析缓存与失效代次（A2 §3）。

缓存只提供存取、索引、代次与失效，不做授权、不接收全局路由、不自行回源；
交付授权与带代次守护的冷读由 ``workspace.resolution`` 负责。
缓存本体只依赖 core，变更订阅者另外依赖 components 的总线机制。
"""

from hivememory.workspace.cache.atom import AtomCache
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.cache.keys import AtomAliasKey, AtomIdKey, ProfileKey
from hivememory.workspace.cache.profile import ProfileCache, ProfileCacheEntry

__all__ = [
    "AtomAliasKey",
    "AtomCache",
    "AtomIdKey",
    "ProfileCache",
    "ProfileCacheEntry",
    "ProfileKey",
    "WorkspaceEpochs",
]
