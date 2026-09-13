"""进程级唯一的 Workspace-oriented 运行时聚合。"""

from __future__ import annotations

import logging
import threading

from hivememory.agent_runtime.aliases.cache import KoakumaAtomCache
from hivememory.agent_runtime.aliases.ports import AtomCachePort
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore

logger = logging.getLogger(__name__)


class WorkspaceRuntime:
    """Workspace-owned working set 及其派生 cache 的进程级聚合。

    职责：
    - 组合并暴露进程内唯一的 ``InMemoryWorkspaceAssetStore`` 与派生 cache
      （当前阶段：Koakuma atom cache；Agent profile cache 随 WRT-3 接入）；
    - 提供幂等的分阶段 ``shutdown()``，只清理派生 cache，不替代消费者各自
      的关闭语义。

    非职责：
    - 不接收可变的"当前 Workspace"，不按 Workspace 创建子 Runtime；
    - 不关闭 ``WorkspaceAssetStore``：``close_and_clear()`` 仍由
      ``HiveMemorySystem.stop()`` 在全部消费者停止之后显式调用；
    - 不承担业务控制、授权、队列、调度或新资源事实源职责。
    """

    def __init__(self) -> None:
        # WorkspaceAsset 是 System-owned working set；聚合内部只创建这一份。
        self._asset_store = InMemoryWorkspaceAssetStore()
        # 派生 cache 由聚合创建并持有所有权；Alice 侧只经 atom_cache_port 注入。
        self._atom_cache = KoakumaAtomCache()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    @property
    def asset_store(self) -> InMemoryWorkspaceAssetStore:
        """进程级唯一的 WorkspaceAssetStore；命令/读取接口保持既有形态。"""
        return self._asset_store

    @property
    def atom_cache_port(self) -> AtomCachePort:
        """供 Alice 侧消费的 atom cache 窄化端口（读写必须携带 Workspace）。"""
        return self._atom_cache

    def shutdown(self) -> None:
        """幂等清理聚合内的派生 cache；不触碰 AssetStore 的关闭语义。

        调用时点必须晚于全部派生 cache 消费者停止（Alice/Patchouli drain
        完成之后）。``clear()`` 自身幂等，因此先清理、后置位，异常重试路径
        重复清理无副作用。
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            cleared_atoms = self._atom_cache.size
            self._atom_cache.clear()
            self._shutdown_done = True
        logger.info(
            "WorkspaceRuntime shutdown 完成：清空派生 atom cache（%s atoms）",
            cleared_atoms,
        )


__all__ = ["WorkspaceRuntime"]
