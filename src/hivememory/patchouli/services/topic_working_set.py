"""Topic 驻留工作集（Working Set）与占用（lease）表。

``TopicWorkingSet`` 是短期话题的驻留管理器：维护有限容量的驻留集合（LRU）、
以 lease 表表达话题占用权，并提供 idle / LRU / shutdown 候选查询。它与
``ShortTermMemoryStore`` 平级协作——Store 持有内容，WorkingSet 只持有索引，
由上层 Familiar 编排，零外部依赖、可纯单元测试。所有方法均为同步方法且不
在内部 await，单事件循环内天然串行；跨线程使用需调用方先行串行化。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from hivememory.core.models import IdentityScope, WorkspaceIdentity, require_identity_scope

# 驻留与 lease 的统一键：占用与驻留是 Workspace 内话题的属性，与某次访问的
# actor 无关——同一 Workspace 可能有多个执行者作用域（不同 agent）。
TopicKey = tuple[WorkspaceIdentity, str]


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """Topic 占用权凭证；调用者持有并在 ``finally`` 里 release。"""

    scope: IdentityScope
    topic_id: str
    acquired_at: float


class TopicWorkingSet:
    """短期话题的驻留工作集与占用表。

    职责：维护 ``max_resident`` 容量的驻留集合与访问顺序（LRU）；以 lease 表
    表达跨 await 占用权（替代记录字段状态机）；提供 idle / LRU / shutdown 候选
    查询与容量判断。不职责：不持有 TopicData 内容（在 Store 里）、不调用
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
        # (workspace, topic_id) -> (最后访问作用域, 访问时间)；OrderedDict 维持
        # LRU 顺序，队首最久未访问，队尾最近访问。
        self._resident: OrderedDict[TopicKey, tuple[IdentityScope, float]] = OrderedDict()
        # (workspace, topic_id) -> LeaseToken
        self._leases: dict[TopicKey, LeaseToken] = {}
        # 只用于相对时长的可注入时钟；默认单调时钟，不受系统墙钟回拨影响。
        self._clock = clock or time.monotonic

    # ========== 驻留追踪 ==========

    def touch(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """标记话题访问：加入或刷新驻留条目，并移到 LRU 队尾。"""
        identity_scope = require_identity_scope(identity_scope)
        key = (identity_scope.workspace_identity, topic_id)
        self._resident[key] = (identity_scope, self._clock())
        self._resident.move_to_end(key)

    def needs_eviction(self, identity_scope: IdentityScope) -> bool:
        """判断该 Workspace 的驻留话题数是否已达容量上限。"""
        identity_scope = require_identity_scope(identity_scope)
        workspace = identity_scope.workspace_identity
        resident_count = sum(1 for key in self._resident if key[0] == workspace)
        return resident_count >= self._max_resident

    def remove(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """从驻留集合移除话题（settle / evict 完成后调用）。

        lease 不自动清理，持有者完成操作后自行 release；目标未驻留时静默忽略。
        """
        identity_scope = require_identity_scope(identity_scope)
        self._resident.pop((identity_scope.workspace_identity, topic_id), None)

    # ========== 候选查询 ==========

    def select_lru_candidate(
        self,
        identity_scope: IdentityScope,
        *,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> str | None:
        """选择同 Workspace 内最久未访问且未被占用的驻留话题。

        ``exclude`` 供调用方在候选失效后改选；返回 ``None`` 表示没有可驱逐候选。
        """
        identity_scope = require_identity_scope(identity_scope)
        workspace = identity_scope.workspace_identity
        for candidate_workspace, topic_id in self._resident:
            if candidate_workspace != workspace or topic_id in exclude:
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
    ) -> list[tuple[IdentityScope, str]]:
        """返回全部 Workspace 中空闲超时且未被占用的驻留话题。

        返回 ``(scope, topic_id)`` 对，scope 为最后访问时冻结的执行作用域，供
        维护路径免重建直接访问 Store；``now`` 可显式指定判定时钟，但须与注入
        时钟同一时基（单调时钟勿传墙钟值）。
        """
        current = now if now is not None else self._clock()
        return [
            (entry_scope, topic_id)
            for (workspace, topic_id), (entry_scope, last_accessed) in self._resident.items()
            if (current - last_accessed) > timeout_seconds
            and (workspace, topic_id) not in self._leases
        ]

    def list_shutdown_candidates(self) -> list[tuple[IdentityScope, str]]:
        """返回全部驻留话题（shutdown 时逐个 settle）；包含正被占用者，由调用方处理。"""
        return [(scope, topic_id) for (_, topic_id), (scope, _) in self._resident.items()]

    # ========== 占用权（lease） ==========

    def acquire(self, identity_scope: IdentityScope, topic_id: str) -> LeaseToken | None:
        """非阻塞获取占用权；已被占用时返回 ``None``；是否驻留不影响获取。"""
        identity_scope = require_identity_scope(identity_scope)
        key = (identity_scope.workspace_identity, topic_id)
        if key in self._leases:
            return None
        lease = LeaseToken(scope=identity_scope, topic_id=topic_id, acquired_at=self._clock())
        self._leases[key] = lease
        return lease

    def release(self, lease: LeaseToken) -> None:
        """释放占用权；令牌已失效（重复释放）时忽略，避免误清后来者的租约。"""
        key = (lease.scope.workspace_identity, lease.topic_id)
        if self._leases.get(key) is lease:
            del self._leases[key]


__all__ = ["LeaseToken", "TopicWorkingSet"]
