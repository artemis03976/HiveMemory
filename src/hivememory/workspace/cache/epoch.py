"""按 Workspace 维护的运行时失效代次（A2 §2.2 / §8.2）。

epoch 是运行时失效代次，不是持久化业务 revision：冷读前记录、回填前比对，
代次变化说明读取期间同一 Workspace 发生过 canonical 变更，旧值不得回填。
按 Workspace 而非按资源维护：alias 冷读前还不知道 memory_id，无法预先记录
资源级代次；Workspace 粒度与失效事件载荷一致，且内存随 Workspace 数有界。
"""

from __future__ import annotations

from hivememory.core.models import WorkspaceIdentity


class WorkspaceEpochs:
    """进程内的 Workspace 失效代次表；推进操作纯内存、不可失败。"""

    def __init__(self) -> None:
        self._epochs: dict[WorkspaceIdentity, int] = {}

    def current(self, workspace: WorkspaceIdentity) -> int:
        """返回 Workspace 当前代次；从未推进过的 Workspace 为 0。"""
        return self._epochs.get(workspace, 0)

    def advance(self, workspace: WorkspaceIdentity) -> int:
        """推进 Workspace 代次并返回新值，使读取期间的在途冷读拒绝回填。"""
        epoch = self._epochs.get(workspace, 0) + 1
        self._epochs[workspace] = epoch
        return epoch


__all__ = ["WorkspaceEpochs"]
