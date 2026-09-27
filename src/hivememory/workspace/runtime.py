"""WorkspaceRuntime：workspace 读取能力与派生缓存的进程内聚合（A2 §1 / 宪章 §3）。

由 System 组合根装配并持有，延续 ``WorkspaceAssetStore`` 的既有归属事实。
聚合失效代次、完整原子缓存、Profile 解析缓存与两个 resolver；L2 冷读端口
（``CanonicalReadBackend``）由组合根注入，本模块不导入总线或 Patchouli 实现。

独立工作（宪章 §4.3）：库不可达时 L1 命中与交付授权照常，L2 冷读显式失败，
不以过期条目伪装新鲜成功。canonical 变更事件的订阅由 A2-2 接入。
"""

from __future__ import annotations

import logging

from hivememory.workspace.cache.atom import AtomCache
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.cache.profile import ProfileCache
from hivememory.workspace.resolution.alias import AliasResolver
from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard
from hivememory.workspace.resolution.profile import ProfileResolver

logger = logging.getLogger(__name__)


class WorkspaceRuntime:
    """workspace 读取能力的运行时聚合：代次、双 cache 与 alias/Profile resolver。"""

    def __init__(
        self,
        *,
        backing: CanonicalReadBackend,
        atom_capacity: int,
        profile_capacity: int,
        max_stale_retries: int = 2,
    ) -> None:
        self._epochs = WorkspaceEpochs()
        self._guard = ColdReadGuard(self._epochs, max_stale_retries=max_stale_retries)
        self._atom_cache = AtomCache(atom_capacity)
        self._profile_cache = ProfileCache(profile_capacity)
        self._aliases = AliasResolver(
            cache=self._atom_cache,
            guard=self._guard,
            backing=backing,
        )
        self._profiles = ProfileResolver(
            cache=self._profile_cache,
            guard=self._guard,
            backing=backing,
        )

    @property
    def aliases(self) -> AliasResolver:
        """canonical alias/UUID 读取与语义检索协作预热。"""
        return self._aliases

    @property
    def profiles(self) -> ProfileResolver:
        """Profile 解析结果缓存与交付授权。"""
        return self._profiles

    @property
    def is_closed(self) -> bool:
        return self._guard.is_closed

    def close(self) -> tuple[int, int]:
        """停止新读，清理仅属于缓存的派生值，返回 (原子条目数, Profile 条目数)。

        在途冷读的结果仍交还原调用方但不再回填；不触碰 Pending、Session 或
        canonical 数据。重复调用返回 ``(0, 0)``。
        """
        if self._guard.is_closed:
            return 0, 0
        self._guard.close()
        atoms = self._atom_cache.clear()
        profiles = self._profile_cache.clear()
        logger.info("WorkspaceRuntime 已关闭（清理 %s atoms, %s profiles）", atoms, profiles)
        return atoms, profiles

    def stats(self) -> dict[str, int]:
        """缓存与冷读的可观测计数（不承诺 Workspace 独立配额）。"""
        return {
            "atom_size": self._atom_cache.size,
            "atom_hits": self._atom_cache.hits,
            "atom_misses": self._atom_cache.misses,
            "atom_evictions": self._atom_cache.evictions,
            "profile_size": self._profile_cache.size,
            "profile_hits": self._profile_cache.hits,
            "profile_misses": self._profile_cache.misses,
            "profile_evictions": self._profile_cache.evictions,
            "cold_reads": self._guard.cold_reads,
            "stale_rejects": self._guard.stale_rejects,
        }


__all__ = ["WorkspaceRuntime"]
