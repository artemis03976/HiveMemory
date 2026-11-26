"""Topic 驻留工作集（Working Set）与占用（lease）表。

``TopicWorkingSet`` 是短期话题的驻留管理器：维护有限容量的驻留集合（LRU）、
以 lease 表表达话题占用权、记下在途沉淀与写进展记录失败的位置，并提供 idle / LRU 候选查询。它与
``ShortTermMemoryStore`` 平级协作——Store 持有内容，WorkingSet 只持有索引，
由上层 Familiar 编排，零外部依赖、可纯单元测试。所有方法均为同步方法且不
在内部 await，单事件循环内天然串行；跨线程使用需调用方先行串行化。
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


class TopicWorkingSet:
    """短期话题的驻留工作集与占用表。

    职责：维护 ``max_resident`` 容量的驻留集合与访问顺序（LRU）；以 lease 表
    表达跨 await 占用权（替代记录字段状态机）；记下驻留话题在途沉淀覆盖到的交互数与写进展记录
    失败的位置；
    提供 idle / LRU 候选查询与容量判断。不职责：不持有 TopicData 内容（在 Store 里）、不调用
    Store / Relay / Queue（零外部依赖）、不解释触发原因（由 Familiar 编排）。
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
        # (workspace, topic_id) -> 访问时间；OrderedDict 维持
        # LRU 顺序，队首最久未访问，队尾最近访问。
        self._resident: OrderedDict[TopicKey, float] = OrderedDict()
        # (workspace, topic_id) -> LeaseToken
        self._leases: dict[TopicKey, LeaseToken] = {}
        # (workspace, topic_id) -> 在途沉淀覆盖到的交互数：只在进程内，随进程退出而消失，
        # 重启后日志停在上一次完成的位置，这段交互照样重新沉淀。
        self._settling: dict[TopicKey, int] = {}
        # (workspace, topic_id) -> 写进展记录失败时那一条要覆盖到的位置：只在进程内，同一位置
        # 不再重试，话题接收新交互之后再写。
        self._progress_entry_failures: dict[TopicKey, int] = {}
        # 只用于相对时长的可注入时钟；默认单调时钟，不受系统墙钟回拨影响。
        self._clock = clock or time.monotonic

    # ========== 驻留追踪 ==========

    def touch(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """标记话题访问：加入或刷新驻留条目，并移到 LRU 队尾。"""
        key = (belong_to, topic_id)
        self._resident[key] = self._clock()
        self._resident.move_to_end(key)

    def needs_eviction(self, belong_to: WorkspaceIdentity) -> bool:
        """判断该 Workspace 的驻留话题数是否已达容量上限。"""
        resident_count = sum(1 for key in self._resident if key[0] == belong_to)
        return resident_count >= self._max_resident

    def remove(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """从驻留集合移除话题（退役 / evict 完成后调用），连同它的在途沉淀与写进展记录失败的位置。

        离开话题池的话题不会再沉淀、也不再写进展记录，二者只用于池中的话题。lease 不自动清理，
        持有者完成操作后自行 release；目标未驻留时静默忽略。
        """
        self._resident.pop((belong_to, topic_id), None)
        self._settling.pop((belong_to, topic_id), None)
        self._progress_entry_failures.pop((belong_to, topic_id), None)

    # ========== 候选查询 ==========

    def select_lru_candidate(
        self,
        belong_to: WorkspaceIdentity,
        *,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> str | None:
        """选择同 Workspace 内最久未访问且未被占用的驻留话题。

        ``exclude`` 供调用方在候选失效后改选；返回 ``None`` 表示没有可驱逐候选。
        """
        for candidate_workspace, topic_id in self._resident:
            if candidate_workspace != belong_to or topic_id in exclude:
                continue
            if (candidate_workspace, topic_id) in self._leases:
                continue  # 跳过正被占用的驻留话题
            return topic_id
        return None

    def list_idle_candidates(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[tuple[WorkspaceIdentity, str]]:
        """返回全部 Workspace 中空闲超时且未被占用的驻留话题。

        返回 ``(belong_to, topic_id)`` 对，后台维护只保留资源归属，不保存
        访问者；``now`` 可显式指定判定时钟，但须与注入
        时钟同一时基（单调时钟勿传墙钟值）。
        """
        current = now if now is not None else self._clock()
        return [
            (workspace, topic_id)
            for (workspace, topic_id), last_accessed in self._resident.items()
            if (current - last_accessed) > timeout_seconds
            and (workspace, topic_id) not in self._leases
        ]

    # ========== 在途沉淀 ==========

    def mark_settling(self, belong_to: WorkspaceIdentity, topic_id: str, count: int) -> None:
        """记下被生成队列接纳、还没完成的沉淀覆盖到的交互数；同一话题只记最远的位置。"""
        key = (belong_to, topic_id)
        self._settling[key] = max(self._settling.get(key, 0), count)

    def settling_count(self, belong_to: WorkspaceIdentity, topic_id: str) -> int:
        """在途沉淀覆盖到的交互数；没有在途沉淀时为 0。"""
        return self._settling.get((belong_to, topic_id), 0)

    def clear_settling(self, belong_to: WorkspaceIdentity, topic_id: str, *, up_to: int) -> None:
        """一次沉淀完成，覆盖到 ``up_to``：不超过它的在途已经完成，清除；更远的仍在途。"""
        key = (belong_to, topic_id)
        if self._settling.get(key, 0) <= up_to:
            self._settling.pop(key, None)

    # ========== 写进展记录失败的位置 ==========

    def mark_progress_entry_failed(
        self, belong_to: WorkspaceIdentity, topic_id: str, covered_until: int
    ) -> None:
        """记下写失败的那一条要覆盖到的位置；话题的交互没有超过它之前不再写。"""
        self._progress_entry_failures[(belong_to, topic_id)] = covered_until

    def progress_entry_failed_until(self, belong_to: WorkspaceIdentity, topic_id: str) -> int:
        """最近一次写进展记录失败时要覆盖到的位置；没有失败过时为 0。"""
        return self._progress_entry_failures.get((belong_to, topic_id), 0)

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


__all__ = ["LeaseToken", "TopicWorkingSet"]
