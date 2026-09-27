"""冷读的失效代次守护与关闭状态（A2 §2.2 / §3.2 / §8.2）。

两个 resolver 共用同一守护：冷读前记录 Workspace 代次，读取返回后比对；
代次变化说明读取期间同一 Workspace 发生过 canonical 变更，旧值既不回填也
不作为成功结果返回，最多重试 ``max_stale_retries`` 次后显式失败。关闭后
拒绝新读；关闭前已发出的冷读结果仍交还原调用方，但不再回填缓存。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import WorkspaceIdentity
from hivememory.workspace.cache.epoch import WorkspaceEpochs

T = TypeVar("T")


class ColdReadGuard:
    """按 Workspace 代次守护冷读回填，并承载 resolver 的关闭状态。"""

    def __init__(self, epochs: WorkspaceEpochs, *, max_stale_retries: int = 2) -> None:
        if max_stale_retries < 0:
            raise ValueError("max_stale_retries 不能为负数")
        self._epochs = epochs
        self._max_stale_retries = max_stale_retries
        self._closed = False
        # 可观测统计：冷读次数与因代次变化拒绝的旧值次数
        self._cold_reads = 0
        self._stale_rejects = 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """停止接收新读；在途冷读完成后不再回填。"""
        self._closed = True

    def ensure_open(self) -> None:
        """关闭后新读显式失败，不以缓存残留或空结果伪装成功。"""
        if self._closed:
            raise ResourceUnavailableError(
                "workspace 读取能力已关闭",
                details={"reason": "workspace_runtime_closed"},
            )

    async def load(
        self,
        workspace: WorkspaceIdentity,
        fetch: Callable[[], Awaitable[T]],
        *,
        retry_on_stale: bool = True,
    ) -> tuple[T, bool]:
        """执行一次受代次守护的冷读，返回 ``(结果, 是否允许回填)``。

        ``retry_on_stale=True``（点读/alias 读取）：代次变化时丢弃结果重读，
        重试耗尽抛 ``ResourceUnavailableError``。``retry_on_stale=False``
        （语义检索的协作预热）：结果照常返回，仅禁止回填。
        """
        for _ in range(self._max_stale_retries + 1):
            self.ensure_open()
            epoch = self._epochs.current(workspace)
            self._cold_reads += 1
            value = await fetch()
            if self._closed:
                return value, False
            if self._epochs.current(workspace) == epoch:
                return value, True
            self._stale_rejects += 1
            if not retry_on_stale:
                return value, False
        raise ResourceUnavailableError(
            "冷读期间 Workspace 持续发生变更，无法确认读取结果的当前性",
            details={
                "reason": "stale_read_retry_exhausted",
                "workspace_id": workspace.workspace_id,
                "attempts": self._max_stale_retries + 1,
            },
        )

    @property
    def cold_reads(self) -> int:
        """累计冷读次数。"""
        return self._cold_reads

    @property
    def stale_rejects(self) -> int:
        """因代次变化拒绝回填的累计次数。"""
        return self._stale_rejects


__all__ = ["ColdReadGuard"]
