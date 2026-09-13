"""进程级唯一的 Workspace-oriented 运行时聚合。"""

from __future__ import annotations

import logging
import threading

from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore

logger = logging.getLogger(__name__)


class WorkspaceRuntime:
    """Workspace-owned working set 及其派生 cache 的进程级聚合。

    职责：
    - 组合并暴露进程内唯一的 ``InMemoryWorkspaceAssetStore`` 与（后续迁移
      阶段接入的）Agent profile cache、Koakuma atom cache 派生缓存；
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
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    @property
    def asset_store(self) -> InMemoryWorkspaceAssetStore:
        """进程级唯一的 WorkspaceAssetStore；命令/读取接口保持既有形态。"""
        return self._asset_store

    def shutdown(self) -> None:
        """幂等清理聚合内的派生状态；不触碰 AssetStore 的关闭语义。

        调用时点必须晚于全部派生 cache 消费者停止（Alice/Patchouli drain
        完成之后），避免清空仍会被读取的状态。当前迁移阶段（WRT-1）聚合
        仅持有 AssetStore；WRT-2/WRT-3 接入两个 cache 后，本方法在同一把
        锁内追加清空步骤。
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        logger.debug("WorkspaceRuntime shutdown 完成（当前无派生 cache 需要清理）")


__all__ = ["WorkspaceRuntime"]
