"""话题池在进程内的状态：每个驻留话题一条记录，加上话题的占用（lease）表。

话题的内容（交互、折叠、进展记录、已沉淀到第几轮）在 ``ShortTermMemoryStore``，随日志落盘；这里
只有进程内才有、进程退出就丢掉的东西。零外部依赖，可纯单元测试。所有方法均为同步方法且不在内部
await，单事件循环内天然串行；跨线程使用需调用方先行串行化。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from hivememory.core.models import WorkspaceIdentity

# 驻留与 lease 的统一键：占用与驻留是 Workspace 内话题的属性，与某次访问的
# actor 无关——同一 Workspace 可能有多个执行者作用域（不同 agent）。
TopicKey = tuple[WorkspaceIdentity, str]


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """Topic 占用权凭证；调用者持有并在 ``finally`` 里 release。"""

    belong_to: WorkspaceIdentity
    topic_id: str
    acquired_at: float


@dataclass(slots=True)
class ResidentTopic:
    """一个驻留话题在进程内的状态，进程退出时全部丢失。

    - ``accessed_at``：最近一次访问的时间（注入时钟的时基），空闲与停顿都按它计时；重启后从
      读回时重新计时。
    - ``settling_until``：被生成队列接纳、还没到终态的沉淀覆盖到第几轮，没有时为 0；重启后
      在途的沉淀已经丢失，日志停在上一次完成的位置，这段交互由空闲扫描重新沉淀。
    - ``progress_failed_until``：最近一次写进展记录失败时要覆盖到第几轮，没有时为 0；话题的
      交互超过它之前不再写，重启后重新尝试。
    """

    accessed_at: float
    settling_until: int = 0
    progress_failed_until: int = 0


class TopicWorkingSet:
    """话题池在进程内的状态：驻留记录与占用表。

    驻留记录（``ResidentTopic``）每个驻留话题一条，按访问顺序排列，最久没用的在前；话题退役或
    删除时这一条整个删除，访问时间、在途沉淀与写进展记录失败的位置一起消失。容量
    ``max_resident`` 按 Workspace 计。

    占用表与驻留记录分开：lease 是跨 await 的占用权，取得它不要求话题驻留（新建的话题在登记
    驻留之前就要写入），话题离开驻留时也不清除，由持有者自己释放。

    不职责：不持有话题内容、不调用存储或队列、不解释触发原因，这些由感知服务编排。
    """

    def __init__(
        self,
        max_resident: int = 5,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_resident < 1:
            raise ValueError("max_resident must be >= 1")
        self._max_resident = max_resident
        # 队首最久未访问，队尾最近访问。
        self._resident: OrderedDict[TopicKey, ResidentTopic] = OrderedDict()
        self._leases: dict[TopicKey, LeaseToken] = {}
        # 只用于相对时长的可注入时钟；默认单调时钟，不受系统墙钟回拨影响。
        self._clock = clock or time.monotonic

    # ========== 驻留 ==========

    def touch(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """标记话题访问：没有驻留时登记一条，已驻留时刷新访问时间；都移到队尾。

        刷新只改访问时间，在途沉淀与写进展记录失败的位置照旧。
        """
        key = (belong_to, topic_id)
        resident = self._resident.get(key)
        if resident is None:
            self._resident[key] = ResidentTopic(accessed_at=self._clock())
        else:
            resident.accessed_at = self._clock()
            self._resident.move_to_end(key)

    def needs_eviction(self, belong_to: WorkspaceIdentity) -> bool:
        """判断该 Workspace 的驻留话题数是否已达容量上限。"""
        resident_count = sum(1 for key in self._resident if key[0] == belong_to)
        return resident_count >= self._max_resident

    def remove(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """话题离开话题池（退役或删除）：删除它的驻留记录；没有驻留时静默忽略。

        lease 不在这里清除，持有者完成操作后自行 release。
        """
        self._resident.pop((belong_to, topic_id), None)

    # ========== 候选查询 ==========

    def select_lru_candidate(
        self,
        belong_to: WorkspaceIdentity,
        *,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> str | None:
        """选择同 Workspace 内最久未访问且未被占用的驻留话题。

        ``exclude`` 供调用方在候选失效后改选；返回 ``None`` 表示没有可退役的候选。
        """
        for candidate_workspace, topic_id in self._resident:
            if candidate_workspace != belong_to or topic_id in exclude:
                continue
            if (candidate_workspace, topic_id) in self._leases:
                continue  # 跳过正被占用的驻留话题
            return topic_id
        return None

    def list_inactive(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[tuple[WorkspaceIdentity, str]]:
        """返回全部 Workspace 中超过 ``timeout_seconds`` 没有访问、且未被占用的驻留话题。

        空闲扫描与停顿扫描共用，阈值不同。返回 ``(belong_to, topic_id)`` 对，后台维护只保留
        资源归属，不保存访问者；``now`` 可显式指定判定时钟，但须与注入时钟同一时基（单调时钟
        勿传墙钟值）。
        """
        current = now if now is not None else self._clock()
        return [
            key
            for key, resident in self._resident.items()
            if (current - resident.accessed_at) > timeout_seconds and key not in self._leases
        ]

    # ========== 在途沉淀与写进展记录失败的位置 ==========
    #
    # 只记在驻留记录上：离开话题池的话题不会再被沉淀或写进展记录，不驻留时什么都不记、读出 0。

    def mark_settling(
        self, belong_to: WorkspaceIdentity, topic_id: str, settling_until: int
    ) -> None:
        """记下被生成队列接纳、还没到终态的沉淀覆盖到第几轮；同一话题只记最远的位置。"""
        resident = self._resident.get((belong_to, topic_id))
        if resident is not None:
            resident.settling_until = max(resident.settling_until, settling_until)

    def settling_until(self, belong_to: WorkspaceIdentity, topic_id: str) -> int:
        """在途沉淀覆盖到第几轮；没有在途沉淀时为 0。"""
        resident = self._resident.get((belong_to, topic_id))
        return 0 if resident is None else resident.settling_until

    def clear_settling(
        self, belong_to: WorkspaceIdentity, topic_id: str, *, settled_until: int
    ) -> None:
        """一次沉淀到达终态，覆盖到 ``settled_until``：不超过它的在途已经结束，清除；更远的仍在途。"""
        resident = self._resident.get((belong_to, topic_id))
        if resident is not None and resident.settling_until <= settled_until:
            resident.settling_until = 0

    def mark_progress_failed(
        self, belong_to: WorkspaceIdentity, topic_id: str, failed_until: int
    ) -> None:
        """记下写失败的那一条要覆盖到第几轮；话题的交互没有超过它之前不再写。"""
        resident = self._resident.get((belong_to, topic_id))
        if resident is not None:
            resident.progress_failed_until = failed_until

    def progress_failed_until(self, belong_to: WorkspaceIdentity, topic_id: str) -> int:
        """最近一次写进展记录失败时要覆盖到第几轮；没有失败过时为 0。"""
        resident = self._resident.get((belong_to, topic_id))
        return 0 if resident is None else resident.progress_failed_until

    # ========== 占用权（lease） ==========

    def acquire(self, belong_to: WorkspaceIdentity, topic_id: str) -> LeaseToken | None:
        """非阻塞获取占用权；已被占用时返回 ``None``；是否驻留不影响获取。"""
        key = (belong_to, topic_id)
        if key in self._leases:
            return None
        lease = LeaseToken(belong_to=belong_to, topic_id=topic_id, acquired_at=self._clock())
        self._leases[key] = lease
        return lease

    def release(self, lease: LeaseToken) -> None:
        """释放占用权；令牌已失效（重复释放）时忽略，避免误清后来者的租约。"""
        key = (lease.belong_to, lease.topic_id)
        if self._leases.get(key) is lease:
            del self._leases[key]


__all__ = ["LeaseToken", "ResidentTopic", "TopicWorkingSet"]
